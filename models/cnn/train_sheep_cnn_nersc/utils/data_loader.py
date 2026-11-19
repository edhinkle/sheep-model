# Modeled after: https://github.com/NERSC/nersc-dl-multigpu/blob/main/utils/data_loader.py
import torch
from torch.utils.data import Subset
from torch.utils.data import DataLoader, Dataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import numpy as np
import sys
import subprocess
import os
import time
import pandas as pd
from scipy import stats
import csv

import sys
sys.path.insert(0, '/global/cfs/cdirs/dune/users/ehinkle/nd_prototypes_ana/sheep-model/models/cnn/spine/src/')
import spine
#from spine.io.dataset.larcv import LArCVDataset
#from spine.io.sample import RandomSequenceBatchSampler, DistributedProxySampler
import yaml
import ROOT


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
try: 
    import h5py
except ImportError:
    install('h5py')
    import h5py

import glob

# Get data loader
def get_data_loader(params, data_location, distributed, train=True, test=False):
    """Function to get the data loader for training/validation/testing."""
    mode = 'train' if train else 'valid' if not test else 'test'
    print("Getting data loader for mode ", mode, " with data location ", data_location)

    dataset = ShowerDataset(params, data_location, mode=mode)

    # Make sure point is in FV if not transforming samples (for testing)
    if dataset._transform_showers == False:
        print("Applying FV cut for untransformed testing/validation samples.")
        keep_indices = dataset._build_fv_subset()
        dataset = Subset(dataset, keep_indices)
        
    batch_size = int(params.local_batch_size if train else params.local_valid_batch_size if not test else params.local_test_batch_size)
    sampler = DistributedSampler(dataset, shuffle=train, drop_last=True) if distributed else None


    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=params.num_data_workers,
                            shuffle=(sampler is None and train), # Don't shuffle validation/test data
                            sampler=sampler,
                            collate_fn=shower_collate_fn,
                            drop_last=True,
                            persistent_workers=False,
                            prefetch_factor=4,
                            pin_memory= torch.cuda.is_available(), # unclear whether this helps ...
                            timeout=600) # timeout to prevent hanging

    return dataloader, sampler

segments_event_data_dtype = np.dtype([
    ('dE', np.float32),  # Energy deposit
    ('x', np.float32),   # X coordinate
    ('y', np.float32),   # Y coordinate
    ('z', np.float32)    # Z coordinate
])

def shower_collate_fn(batch): 
    """Collate function to combine multiple samples into a batch."""

    data_list = []
    labels_list = []
    ve_frac_list = []
    mg_frac_list = []
    oob_frac_list = []
    start_pos_list = []
    rot_mat_list = []
    idx_list = []
    minE_list = []
    maxE_list = []
    meanE_list = []
    medE_list = []
    numVox_list = []

    for batch_idx, (data, label, VE_frac, MG_frac, OOB_frac, start_pos, rot_mat, idx, minE, maxE, meanE, medE, numVox) in enumerate(batch):
        # Set batch index
        batch_data = data.clone()

        batch_data[:, 0] = batch_idx  # Set batch index to the first column

        data_list.append(batch_data)

        labels_list.append(label)
        ve_frac_list.append(VE_frac)
        mg_frac_list.append(MG_frac)
        oob_frac_list.append(OOB_frac)
        start_pos_list.append(start_pos)
        rot_mat_list.append(rot_mat)
        idx_list.append(idx)
        minE_list.append(minE)
        maxE_list.append(maxE)
        meanE_list.append(meanE)
        medE_list.append(medE)
        numVox_list.append(numVox)

    # Concatenate all coordinates and features into single tensors
    batched_data = torch.cat(data_list, dim=0)
    batched_labels = torch.stack(labels_list, dim=0).reshape(-1, 1) # ESSENTIAL -- ensures output and labels are same shape
    batched_ve_frac = torch.stack(ve_frac_list, dim=0).reshape(-1, 1)
    batched_mg_frac = torch.stack(mg_frac_list, dim=0).reshape(-1, 1)
    batched_oob_frac = torch.stack(oob_frac_list, dim=0).reshape(-1, 1)
    batched_start_pos = torch.stack(start_pos_list, dim=0) # Shape (B, 3)
    batched_rot_mat = torch.stack(rot_mat_list, dim=0) # Shape (B, 3, 3)
    batched_idx = torch.stack(idx_list, dim=0).reshape(-1, 1) 
    batched_minE = torch.stack(minE_list, dim=0).reshape(-1, 1) 
    batched_maxE = torch.stack(maxE_list, dim=0).reshape(-1, 1) 
    batched_meanE = torch.stack(meanE_list, dim=0).reshape(-1, 1) 
    batched_medE = torch.stack(medE_list, dim=0).reshape(-1, 1) 
    batched_numVox = torch.stack(numVox_list, dim=0).reshape(-1, 1) 

    #print("Batched data:", batched_idx)
    #print("Batched labels:", batched_labels)

    return batched_data, batched_labels, batched_ve_frac, batched_mg_frac, batched_oob_frac, batched_start_pos, batched_rot_mat, \
           batched_idx, batched_minE, batched_maxE, batched_meanE, batched_medE, batched_numVox


class ShowerDataset(Dataset):
         
    def __init__(self, params, data_location, mode='train'):

        """
        Args:
            params:                Parameters from yaml file
            data_location:         Path to dataset directory
            mode:                  'train' | 'valid' | 'test' 
        """
        self._debug = params.debug

        # Set mode and training parameters
        self.mode = mode
        self.num_workers = params.num_data_workers
        self._train_logE = params.train_logE  # Whether to take log of energy for training targets (for better training stability)
        self._transform_showers = params.transform_showers  # Whether to transform input showers or not

        # Set num_files for train/validation/test based on mode
        self._file_dir = data_location
        self._num_files_train = params.train_files
        self._num_files_val = params.val_files
        self._num_files_test = params.test_files
        self._set_dataset_file_list()  # Get list of files in dataset directory
        self._set_events_per_file()  # Get number of events per file + file indices

        # Photons vs. electrons
        self._photons_in_sample = params.photons_in_sample
        self._electrons_in_sample = params.electrons_in_sample
        self._event_vertex_in_fv = params.event_vertex_in_fv
        self._particle_start_in_fv = params.particle_start_in_fv

        # Set geometry 
        self._detector_active_regions = np.array(params.detector_active_regions)

        # Get min and max bounds for each detector module
        self._min_bounds = self._detector_active_regions[:, 0, :] # Shape (M, 3) where M is the number of detector modules/active volumes
        self._max_bounds = self._detector_active_regions[:, 1, :] # Shape (M, 3) where M is the number of detector modules/active volumes

        # Set min/max coordinates and number of voxels
        self._min_xyz = np.array([np.min(self._detector_active_regions[:, :, 0]), 
                                  np.min(self._detector_active_regions[:, :, 1]),
                                  np.min(self._detector_active_regions[:, :, 2])])
        self._max_xyz = np.array([np.max(self._detector_active_regions[:, :, 0]),
                                  np.max(self._detector_active_regions[:, :, 1]),
                                  np.max(self._detector_active_regions[:, :, 2])])
        self._voxel_size = np.array(params.voxel_size)
        self._num_voxels = np.ceil((self._max_xyz - self._min_xyz) / self._voxel_size).astype(int)

        # Set fiducial volumes
        self._inner_wall_fv_cut = params.inner_wall_fv_cut  # Fiducial volume cut in cm on inner walls (modules)
        self._full_fv_cut = params.full_fv_cut # Turn on outer and downstream wall "extra FV cuts"
        self._outer_wall_fv_cut = params.outer_wall_fv_cut # Extra FV cut -- outer walls of full detector
        self._ds_wall_fv_cut = params.ds_wall_fv_cut # Extra FV cut -- downstream wall in z

        # Set energy thresholds -- minimum visible and whether to apply thresholding
        self._min_visible_energy = params.min_visible_energy  # For numerical stability in MinkowskiEngine (Minimum energy target) 
        self._set_energy_threshold = params.set_energy_threshold  # Only look at deposits above threshold?
        self._energy_threshold = params.energy_threshold  # Only look at deposits above threshold

        # Set random seed and related variables for reproducibility
        self._RANDOM_SEED = int(params.random_seed)
        self._epoch = 0

        # Set transformation mode and fixed parameters for visualization
        self._fixed_transformation_mode = False
        self._fixed_start_pos = None
        self._fixed_rotation_matrix = None
        self._last_fixed_transformed_segments = None
        self._last_fixed_all_energy_points = None
        self._last_fixed_visible_points_mask = None

        if self.debug == True:
            print("Number of files:", len(self._file_list))
            print("File list:",self._file_list)
            print("Events per file:", self._events_per_file)
            print(f"Max coordinates: {self._max_xyz}")
            print(f"Min coordinates: {self._min_xyz}")

    def __len__(self):
        return np.sum(self._events_per_file)

    def __getitem__(self, idx):

        file_idx, event_local_idx = self._decode_idx(idx)  # Decode the global index into file and event indices
        h5_file_name = self._file_list[file_idx]

        if self._debug == True:
            print(f"Loading file: {h5_file_name}, File index: {file_idx}, Event local index: {event_local_idx}")  # Debugging line to check which file and event is being loaded
            print(f"Event global index: {idx}")  # Debugging line to check global index being loaded
        
        with h5py.File(h5_file_name, 'r') as h5_file:
            true_KE_initial = float(h5_file['ke_initial'][event_local_idx])  # Access the true KE initial for the event
            if self._debug == True:
                print(f"File index: {file_idx}, Event local index: {event_local_idx}, True KE Initial: {true_KE_initial}")
            try:
                particle_start = np.array(h5_file['start_points'][event_local_idx])  # Access the start point for the event
            except:
                particle_start = np.array([0, 0, 0], dtype=np.float32)  # Default to origin if start point is not available

            # Get start and end voxels
            voxels_flat = h5_file['voxels_flat']  # Access the flat voxel array
            voxels_start = int(h5_file['voxels_offsets'][event_local_idx])
            voxels_end = int(h5_file['voxels_offsets'][event_local_idx + 1])
            event_voxels = voxels_flat[voxels_start:voxels_end]  # Get the voxel array for the event
            positions_cm = np.array(event_voxels[:, :3])  # Get the positions in cm
            energy_per_voxel = np.array(event_voxels[:, 3])  # Get the energy per voxel

        # Same for train/validation/test now
        rng = self._get_deterministic_rng_per_event(idx)
        if self._debug == True:
            print(f"Event ID: {idx}, RNG state: {rng.bit_generator.state}")  # Debugging line to check RNG state for each event

        # Get filtered segments based on whether transformation is enabled or not
        if self._transform_showers == True:
            # Use fixed transformation if visualization mode is enabled
            if self._fixed_transformation_mode and self._fixed_start_pos is not None:
                vis_segs, ve_frac, mg_frac, oob_frac, start_pos, rot_mat = self._get_filtered_segments(idx,
                rng, positions_cm, energy_per_voxel, true_KE_initial, particle_start, self._min_visible_energy,
                fixed_start_pos=self._fixed_start_pos, fixed_rotation_matrix=self._fixed_rotation_matrix
            )
            else:
                vis_segs, ve_frac, mg_frac, oob_frac, start_pos, rot_mat = self._get_filtered_segments(idx,
                rng, positions_cm, energy_per_voxel, true_KE_initial, particle_start, self._min_visible_energy
            )
        else:
            # No transformation -- just use original positions and energy
            vis_segs = np.zeros(positions_cm.shape[0], dtype=segments_event_data_dtype)  # Create an empty array for visible segments
            vis_segs['dE'] = energy_per_voxel.flatten()  # Fill the energy depositions
            vis_segs['x'] = positions_cm[:, 0]  # Fill the x coordinate
            vis_segs['y'] = positions_cm[:, 1]  # Fill the y coordinate
            vis_segs['z'] = positions_cm[:, 2]  # Fill the z coordinate

            ve_frac = np.sum(energy_per_voxel) / true_KE_initial  # Calculate visible energy fraction
            mg_frac = 0.0  # No module gap energy fraction since no transformation
            oob_frac = 0.0  # No out-of-bounds energy fraction since no transformation
            start_pos = particle_start  # Use the original particle start position
            rot_mat = np.eye(3)  # Identity matrix for rotation (no rotation)

        # Voxelize the filtered segments SPARSELY
        coords, features = self._voxelize_sparse(vis_segs)

        # Apply additional augmentation if enabled (e.g. energy thresholds, smearing, etc.)
        if self._set_energy_threshold:
            # Apply energy threshold to features and coords
            if self._debug == True:
                print(f"Applying energy threshold of {self._energy_threshold} MeV to features. Number of voxels before thresholding: {len(coords)}")
            energy_mask = features[:, 0] >= self._energy_threshold
            coords = coords[energy_mask]
            features = features[energy_mask]
            total_visible_energy_after_threshold = np.sum(features[:, 0])  # Sum of energies in MeV after thresholding
            ve_frac = total_visible_energy_after_threshold / true_KE_initial if true_KE_initial > 0 else 0.0
            if self._debug == True:
                print(f"Applied energy threshold of {self._energy_threshold} MeV. Number of voxels after thresholding: {len(coords)}")

        # Compute metrics on filled voxels
        min_voxel_E = np.min(features)
        max_voxel_E = np.max(features)
        mean_voxel_E = np.mean(features)
        median_voxel_E = np.median(features)
        num_voxels_filled = len(features)
        
        # Convert coords, features, labels, to PyTorch tensors
        coords = torch.from_numpy(coords).contiguous()  # Keep as int32 for voxel indices
        features = torch.from_numpy(features).contiguous()  # Already float32
        combined_data = torch.cat([coords.float(), features], dim=1)  # Convert coords to float only for concatenation

        # Instead of converting to GeV, take log of energy to target output range
        true_KE_initial_tensor = torch.tensor(true_KE_initial, dtype=torch.float32) 
        if self._train_logE == True:
            true_KE_initial_tensor = torch.tensor(true_KE_initial, dtype=torch.float32) 
            true_KE_initial_tensor = torch.log(true_KE_initial_tensor)  
        else:
            true_KE_initial_tensor = torch.tensor(true_KE_initial / 1000.0, dtype=torch.float32) # Convert to GeV
        
        # Convert metrics and identifiers to torch tensors
        ve_frac_tensor = torch.tensor(ve_frac, dtype=torch.float32)
        mg_frac_tensor = torch.tensor(mg_frac, dtype=torch.float32)
        oob_frac_tensor = torch.tensor(oob_frac, dtype=torch.float32)
        start_pos_tensor = torch.tensor(start_pos, dtype=torch.float32)
        rot_mat_tensor = torch.tensor(rot_mat, dtype=torch.float32)
        idx_tensor = torch.tensor(idx, dtype=torch.int32)
        min_voxel_E_tensor = torch.tensor(min_voxel_E, dtype=torch.float32)
        max_voxel_E_tensor = torch.tensor(max_voxel_E, dtype=torch.float32)
        mean_voxel_E_tensor = torch.tensor(mean_voxel_E, dtype=torch.float32)
        median_voxel_E_tensor = torch.tensor(median_voxel_E, dtype=torch.float32)
        num_voxels_filled_tensor = torch.tensor(num_voxels_filled, dtype=torch.int32)
        
        if self._debug == True:
            print("Start point:", start_pos_tensor)
            print("Rotation matrix:", rot_mat_tensor)
            print("True KE:", true_KE_initial_tensor)

        return combined_data, true_KE_initial_tensor, ve_frac_tensor, mg_frac_tensor, oob_frac_tensor, start_pos_tensor, rot_mat_tensor, idx_tensor, min_voxel_E_tensor, max_voxel_E_tensor, mean_voxel_E_tensor, median_voxel_E_tensor, num_voxels_filled_tensor


    # Method to get file_idx, event_idx pair from global idx
    def _decode_idx(self, idx):
        """Decode a global index into a file index and an event index."""
        file_idx = np.digitize(idx, self._event_total_by_file) - 1  # Find the file index
        if self._debug == True:
            print("File index:", file_idx)  # Debugging line to check file index
        event_idx = idx - self._event_total_by_file[file_idx]  # Find the event index within that file
        return file_idx, event_idx
    
    # Allow for fixed transformation parameters for debugging
    def set_visualization_transformation(self, start_pos, rotation_matrix):
        """Set fixed transformation parameters for visualization. Call with None to reset."""
        if start_pos is None or rotation_matrix is None:
            self._fixed_transformation_mode = False
            self._fixed_start_pos = None
            self._fixed_rotation_matrix = None
            self._last_fixed_transformation_transformed_segments = None
            self._last_fixed_transformation_all_energy_points = None
            self._last_fixed_transformation_visible_points_mask = None
        else:
            self._fixed_transformation_mode = True
            self._fixed_start_pos = np.asarray(start_pos)
            self._fixed_rotation_matrix = np.asarray(rotation_matrix)
            if self._debug == True:
                print("Setting fixed transformation: start_pos={}, rotation_matrix={}".format(self._fixed_start_pos, self._fixed_rotation_matrix))

    # Set deterministic rng per event for validation/testing
    def _get_deterministic_rng_per_event(self, event_id: int):
        """Get a deterministic random number generator for a given event and epoch for determinstic poses."""
        if self.mode == 'train' or self.mode == 'test':
            if self._debug == True:
                print("Using deterministic RNG for event {} in mode {} with epoch {}".format(event_id, self.mode, self._epoch))
            mixed_seed = (self._RANDOM_SEED * 1_000_0003) ^ (int(event_id) * 97) ^ (int(self._epoch) * 1_000_000 + 1337)
        elif self.mode == 'valid':
            mixed_seed = (self._RANDOM_SEED * 1_000_0003) ^ (int(event_id) * 97)
        return np.random.default_rng(np.uint64(mixed_seed & 0xFFFFFFFFFFFFFFFF))  # Ensure seed fits in uint64

    # Method to convert random start position and start direction to set of filtered visible depositions
    def _get_filtered_segments(self, idx, rng, positions, energy_per_larcv_voxel, true_KE_initial, particle_start=np.array([0,0,0]),
                               min_visible_energy=5.0, fixed_start_pos=None, fixed_rotation_matrix=None):
        ''' Method to convert random start position and start direction to set of filtered visible depositions
        Inputs:
            - rng: random number generator (different for train/val/test for reproducibility)
            - positions: array of voxel x, y, z positions (cm)
            - energy_per_larcv_voxel: array of energy values for each voxel (MeV)
            - true_KE_initial: initial kinetic energy of the shower (for calculating visible energy fraction)
            - min_visible_energy: minimum visible energy required (in MeV)
            - particle_start: the start point for the event (mostly relevant for photons bc first energy deposition may not be at event vtx)
            - fixed_start_pos: optional fixed start position for visualization (overrides random sampling if provided)
            - fixed_rotation_matrix: optional fixed rotation matrix for visualization (overrides random sampling if provided
        Outputs:
            - transformed segments: array of segments with transformed positions
            - in_any_volume: boolean array indicating whether each segment is within any detector volume
            - start_pos: the sampled start position
            - rotation_matrix: the sampled rotation matrix
        '''
        
        # Sample a random start position and rotation matrix until the visible energy fraction is above the minimum threshold
        # This ensures that the sampled shower has enough visible energy depositions in the detector volumes
        # TO-DO: Figure out how to sample start position and rotation matrix such that the visible energy fraction is above the minimum threshold more often on first try

        # If fixed transformation parameters provided, skip random sampling
        if fixed_start_pos is not None and fixed_rotation_matrix is not None:
            start_pos = fixed_start_pos
            rotation_matrix = fixed_rotation_matrix
            transformed_segments_xyz = np.einsum('ij, kj->ki', rotation_matrix, positions) + start_pos
            in_any_volume = np.any(
                np.all((transformed_segments_xyz[:, None, :] >= self._min_bounds) &
                       (transformed_segments_xyz[:, None, :] <= self._max_bounds), axis=2),
                axis=1
            )
            visible_energy = np.sum(energy_per_larcv_voxel[in_any_volume])
            self._last_fixed_transformation_transformed_segments = transformed_segments_xyz
            self._last_fixed_transformation_all_energy_points = energy_per_larcv_voxel
            self._last_fixed_transformation_visible_points_mask = in_any_volume
        else:
            max_VE_under_threshold = 1.
            best_transformed_segments = positions
            best_in_any_volume = np.any(np.all((best_transformed_segments[:, None, :] >= self._min_bounds) &
                               (best_transformed_segments[:, None, :] <= self._max_bounds), axis=2),
                                axis=1)
            positions_to_use = positions
            try:
                iteration = 0    
                for _ in range(5000):
                    iteration += 1
                    start_pos = self._sample_random_start_position(rng)  # Sample a random start position
                    rotation_matrix = self._sample_random_rotation_matrix(rng, true_KE_initial)  # Sample a random rotation matrix
                    #print("Rotation matrix:\n", rotation_matrix)  # Debugging line to check rotation matrix
                    #print("A few segment positions:", segment_positions[:5])  # Debugging line to check segment positions
                    #transformed_segments_xyz = segment_positions @ rotation_matrix.T + start_pos # Shape (N, 3) where N is the number of segments
                    # TODO: Fix sampling here so that it is possible to have both event vertex and particle start in FV, or particle start not in FV, and still get visible energy fraction above threshold
                    if self._event_vertex_in_fv == True and self._particle_start_in_fv == True:
                        transformed_start = np.einsum('ij, j->i', rotation_matrix, particle_start) + start_pos  # Transform the particle start position
                        if self._is_point_in_fv(transformed_start) == False:
                            continue  # Skip this iteration if the transformed start position is not in the FV
                        else:
                            positions_to_use = positions # don't shift positions, keep them relative to event vertex
                    elif self._event_vertex_in_fv == True and self._particle_start_in_fv == False:
                        positions_to_use = positions # don't shift positions, keep them relative to event vertex
                    elif self._event_vertex_in_fv == False and self._particle_start_in_fv == True:
                        #print("Only particle start in FV is true. Shifting start points.")
                        positions_to_use = positions - particle_start  # Shift positions to be relative to particle start
                    else: 
                        raise ValueError("Both event_vertex_in_fv and particle_start_in_fv cannot be False. Please check your config file.")
                    transformed_segments_xyz = np.einsum('ij, kj->ki', rotation_matrix, positions_to_use) + start_pos  # Efficient matrix multiplication and addition

                    # Check if each segment is within min or max bounds for each dimension
                    # Then, check if xyz of segment is within min and max bounds for any volume
                    # This is done by checking if the segment is within the bounds of any detector module
                    in_any_volume = np.any(
                        np.all((transformed_segments_xyz[:, None, :] >= self._min_bounds) &
                               (transformed_segments_xyz[:, None, :] <= self._max_bounds), axis=2),
                        axis=1
                    )

                    # Check visible energy depositions
                    visible_energy = np.sum(energy_per_larcv_voxel[in_any_volume])  # Sum the energy depositions of visible segments

                    #if visible_energy_fraction >= min_visible_energy:
                    if visible_energy >= self._min_visible_energy:
                        break
                    elif visible_energy > max_VE_under_threshold:
                        max_VE_under_threshold = visible_energy
                        best_transformed_segments = transformed_segments_xyz
                        best_in_any_volume = in_any_volume
                        best_start_pos = start_pos
                        best_rotation_matrix = rotation_matrix
                    else: continue
            except:
                print(f"Resampling timed out. Best visible energy achieved was {max_VE_under_threshold} and will be used.", flush=True)
                visible_energy = max_VE_under_threshold
                transformed_segments_xyz = best_transformed_segments
                in_any_volume = best_in_any_volume
                start_pos = best_start_pos
                rotation_matrix = best_rotation_matrix

        # Get visible energy fraction, module gap energy fraction, and uncontained energy fraction
        # visible_energy_fraction is calculated above

        # Module gap energy 
        in_abs_det_bounds = np.all((transformed_segments_xyz[:, :] >= self._min_xyz) &
                                   (transformed_segments_xyz[:, :] <= self._max_xyz), axis=1)
        in_mod_gaps = in_abs_det_bounds & ~in_any_volume
        module_gap_energy = np.sum(energy_per_larcv_voxel[in_mod_gaps])

        # Uncontained energy 
        out_of_detector = ~in_abs_det_bounds
        out_of_det_bounds_energy = np.sum(energy_per_larcv_voxel[out_of_detector])

        # Calculate energy fractions
        visible_energy_fraction = visible_energy / true_KE_initial  # Calculate the fraction of visible energy compared to the initial kinetic energy
        module_gap_energy_fraction = module_gap_energy / true_KE_initial
        out_of_det_bounds_energy_fraction = out_of_det_bounds_energy / true_KE_initial

        if self._debug == True:
            print("Visible Energy Fraction:", visible_energy_fraction)
            print("Module Gap Energy Fraction:", module_gap_energy_fraction)
            print("Out of Bounds Energy Fraction:", out_of_det_bounds_energy_fraction)
            print("Sum of Energy Fractions:", visible_energy_fraction+module_gap_energy_fraction+out_of_det_bounds_energy_fraction)

        if visible_energy <= 0.:
            file_idx, event_local_idx = self._decode_idx(idx)
            print("Positions minus positions to use:", positions - positions_to_use)
            print('Iteration: ', iteration)
            print("File: ", self._file_list[file_idx])
            print("Event local index: ", event_local_idx)
            print("True KE Initial: ", true_KE_initial)
            print("Visible Energy: ", visible_energy)
            print("Start position: ", start_pos)
            print("Rotation matrix: ", rotation_matrix)
            print("Positions to use in visible volume: ", positions_to_use[in_any_volume])
            print("Module gap energy fraction: ", module_gap_energy_fraction)
            print("Out of bounds energy fraction: ", out_of_det_bounds_energy_fraction)
            print("Min bounds: ", self._min_bounds)
            print("Max bounds: ", self._max_bounds)
 
        visible_xyz = transformed_segments_xyz[in_any_volume]  # Get the positions of visible segments
        visible_dE = energy_per_larcv_voxel[in_any_volume]  # Get the energy
        visible_segments = np.zeros(visible_xyz.shape[0], dtype=segments_event_data_dtype)  # Create an empty array for visible segments
        visible_segments['dE'] = visible_dE.flatten()  # Fill the energy depositions
        visible_segments['x'] = visible_xyz[:, 0]  # Fill the x coordinate
        visible_segments['y'] = visible_xyz[:, 1]  # Fill the y coordinate
        visible_segments['z'] = visible_xyz[:, 2]  # Fill the z coordinate

        return visible_segments, visible_energy_fraction, module_gap_energy_fraction, out_of_det_bounds_energy_fraction, start_pos, rotation_matrix

  
    # Method to get uniformly randomly sampled directions
    def _sample_random_rotation_matrix(self, rng, true_KE_initial=None):
        """Generate a random rotation matrix -- rng depends on train/val/test mode for reproducibility"""

        # Step 1: Uniform spherical sampling (use z->theta to reduce oversampling at poles) 
        phi = rng.uniform(0, 2 * np.pi)
        z = rng.uniform(-1, 1)
        theta = np.arccos(z) 

        # Step 2: Get direction vector
        # TO-DO: Add back way to sample from restricted range based on energy, angle
        sin_theta = np.sin(theta)
        direction = np.array([
            sin_theta * np.cos(phi),
            sin_theta * np.sin(phi),
            z
        ])

        # Step 3: Get random "roll" angle (rotation around direction vector axis)
        psi = rng.uniform(0, 2 * np.pi)

        # Step 4: Create orthonormal basis using direction vector and two additional vectors
        v = np.array([1, 0, 0]) if abs(direction[0]) < 0.99 else np.array([0, 1, 0])

        u1 = np.cross(v, direction)
        u1 /= np.linalg.norm(u1)
        u2 = np.cross(direction, u1)

        # Step 5: Rotate u1 and u2 basis vectors by roll angle (rotating around direction vector)
        b1 = np.cos(psi) * u1 + np.sin(psi) * u2
        b2 = -np.sin(psi) * u1 + np.cos(psi) * u2
        d = direction # not rotated because it is the axis of rotation

        # Step 6: Create rotation matrix
        rotation_matrix = np.column_stack((b1, b2, d)) # in SO(3)

        return rotation_matrix
    
    # Method to get random start position
    def _sample_random_start_position(self, rng):
        """Sample a random start position within the given range -- rng depends on train/val/test mode for reproducibility"""

        # Only set FV with inner wall cuts
        if self._full_fv_cut == False:
            # restrict start position to the detector active regions
            num_modules = len(self._detector_active_regions)
            module_idx = rng.integers(0, num_modules)  # Randomly select a detector module
            x_start = rng.uniform(self._detector_active_regions[module_idx, 0, 0]+self._inner_wall_fv_cut, 
                                  self._detector_active_regions[module_idx, 1, 0]-self._inner_wall_fv_cut)
            y_start = rng.uniform(self._detector_active_regions[module_idx, 0, 1]+self._inner_wall_fv_cut, 
                                  self._detector_active_regions[module_idx, 1, 1]-self._inner_wall_fv_cut)
            z_start = rng.uniform(self._detector_active_regions[module_idx, 0, 2]+self._inner_wall_fv_cut, 
                                  self._detector_active_regions[module_idx, 1, 2]-self._inner_wall_fv_cut)
        # Add additional FV cuts for outer walls and downstream wall
        else:   
            try: 
                for _ in range(5000):
                    # restrict start position to the detector active regions
                    num_modules = len(self._detector_active_regions)
                    module_idx = rng.integers(0, num_modules)  # Randomly select a detector module
                    x_start = rng.uniform(self._detector_active_regions[module_idx, 0, 0]+self._inner_wall_fv_cut, 
                                          self._detector_active_regions[module_idx, 1, 0]-self._inner_wall_fv_cut)
                    y_start = rng.uniform(self._detector_active_regions[module_idx, 0, 1]+self._inner_wall_fv_cut, 
                                          self._detector_active_regions[module_idx, 1, 1]-self._inner_wall_fv_cut)
                    z_start = rng.uniform(self._detector_active_regions[module_idx, 0, 2]+self._inner_wall_fv_cut, 
                                          self._detector_active_regions[module_idx, 1, 2]-self._inner_wall_fv_cut)


                    if (z_start > self._max_xyz[2]-self._ds_wall_fv_cut): continue
                    elif (z_start < self._min_xyz[2]+self._outer_wall_fv_cut): continue
                    elif (x_start > self._max_xyz[0]-self._outer_wall_fv_cut): continue
                    elif (x_start < self._min_xyz[0]+self._outer_wall_fv_cut): continue
                    elif (y_start > self._max_xyz[1]-self._outer_wall_fv_cut): continue
                    elif (y_start < self._min_xyz[1]+self._outer_wall_fv_cut): continue
                    else: break
            except:
                print(f"Resampling timed out. Will use last sampled point, ({x_start}, {y_start}, {z_start}).", flush=True)

        if self._debug == True:
            print("Sampled start position: ", (x_start, y_start, z_start))
        return np.array([x_start, y_start, z_start])
    
    # Method to check if point is in fiducial volume
    def _is_point_in_fv(self, point):
        """Check if a point is within the fiducial volume defined by detector active regions and cuts."""

        # Check if point is within any of the detector active regions with inner wall cut applied
        in_any_module_fv = np.any(np.all((point >= self._detector_active_regions[:, 0, :]+self._inner_wall_fv_cut) &
                              (point <= self._detector_active_regions[:, 1, :]-self._inner_wall_fv_cut), axis=1))
        
        if self._full_fv_cut == False:
            # Only FV cut is same for all sides (inner wall)
            return in_any_module_fv
        else:
            # Check if point is within the fiducial volume defined by the outer wall and downstream wall cuts
            in_fv = (point[0] > self._min_xyz[0]+self._outer_wall_fv_cut) and \
                    (point[0] < self._max_xyz[0]-self._outer_wall_fv_cut) and \
                    (point[1] > self._min_xyz[1]+self._outer_wall_fv_cut) and \
                    (point[1] < self._max_xyz[1]-self._outer_wall_fv_cut) and \
                    (point[2] > self._min_xyz[2]+self._outer_wall_fv_cut) and \
                    (point[2] < self._max_xyz[2]-self._ds_wall_fv_cut)
            return in_any_module_fv and in_fv

    # Method to build subset of points in fv
    def _build_fv_subset(self):

        if self._transform_showers == True:
            print("Transforming showers. FV is already enforced in the transformation.")
            return None
        
        keep_indices = []

        for file_idx, file_name in enumerate(self._file_list):
            with h5py.File(file_name, "r") as f:
                start_points = np.asarray(f["start_points"])
                n_events = start_points.shape[0]

                for event_local_idx in range(n_events):
                    start_pos = start_points[event_local_idx]
                    if self._is_point_in_fv(start_pos):
                        global_idx = self._event_total_by_file[file_idx] + event_local_idx
                        keep_indices.append(global_idx)

        return keep_indices

    # Method to get list of files in dataset directory
    def _set_dataset_file_list(self):
        """Get list of files in dataset directory"""
        self._file_list = []

        stop_at = -1
        if self.mode == 'train':
            stop_at = self._num_files_train
        elif self.mode == 'valid':
            stop_at = self._num_files_val
        elif self.mode == 'test':            
            stop_at = self._num_files_test

        for file in sorted(glob.glob(self._file_dir + '*.h*5')): 
            self._file_list.append(file)
            if len(self._file_list) == stop_at:
                break
        
        if len(self._file_list) == 0:
            raise ValueError("No files found in dataset directory: {}".format(self._file_dir)) 
        
    # Set epoch method to update epoch count for deterministic RNGs
    def _set_epoch(self, epoch: int):
         """Called from training loop each epoch so per-sample RNGs can vary deterministically per epoch."""
         self._epoch = int(epoch)
        
    # Method to get number of events per file
    def _set_events_per_file(self):
        self._events_per_file = []
        for file_name in self._file_list:
            with h5py.File(file_name, 'r') as f:
                self._events_per_file.append(int(f.attrs['n_events']))
        self._events_per_file = np.array(self._events_per_file)
        self._event_total_by_file = np.cumsum(self._events_per_file)  # Cumulative sum to get event indices
        self._event_total_by_file = np.insert(self._event_total_by_file, 0, 0)

    def _unique_voxel_indices(self, coords, features):
        """Get unique voxel indices and their corresponding features by combining
        features at the same voxel index."""

        if len(coords) <= 1:
            return coords, features # nothing to combine
        if self._debug == True:
            print("Coordinates before combining duplicates:", coords[:10])
        # Find unique coordinates and aggregate features
        #  (use np.void for structured array/bytewise comparison vs. elementwise)
        coord_view = coords.view(np.dtype((np.void, coords.dtype.itemsize * coords.shape[1])))
        unique_coords, inverse_indices = np.unique(coord_view, return_inverse=True)

        # Aggregate features at the same voxel index
        unique_features = np.zeros((len(unique_coords), 1), dtype=features.dtype)
        for i in range(len(unique_coords)):
            indices_mask = (inverse_indices == i)
            if self._debug == True and sum(indices_mask) > 1:
                print(f"Combining {sum(indices_mask)} features at the same voxel index with energies {features[indices_mask]}.")  # Debugging line to check when duplicates are being combined
            unique_features[i] = np.sum(features[indices_mask], axis=0)  # Sum features at the same voxel index

        # Convert unique_coords back to original dtype
        if self._debug == True:
            print("Unique coordinates after combining duplicates:", unique_coords[:10])
        unique_coords = unique_coords.view(coords.dtype).reshape(-1, coords.shape[1])

        # Return unique coordinates and combined features
        return unique_coords, unique_features

    def _voxelize_sparse(self, segments):

        # Base case -- no segments 
        if len(segments) == 0:
            return np.zeros((0, 4), dtype=np.int32), np.zeros((0, 1), dtype=np.float32)
        
        # Get the voxel indices for each segment
        seg_pos = np.stack([segments[a] for a in ['x','y','z']], axis=1)
        voxel_indices = ((seg_pos - self._min_xyz) // self._voxel_size).astype(np.int32)

        # Add dimension for batching (need for MinkowskiEngine
        coords = np.zeros((len(voxel_indices), 4), dtype=np.int32)
        coords[:, 0] = 0 # Batch index (0 for single batch)
        coords[:, 1:] = voxel_indices # x, y, z voxel indices
        
        # Get the voxel values (energy depositions) - make it 2D to match coords
        features = np.array(segments['dE'], dtype=np.float32).reshape(-1, 1)
        if self._debug == True:
            print("Features before combining duplicates:", features[:5])

        return self._unique_voxel_indices(coords, features)
    
   
