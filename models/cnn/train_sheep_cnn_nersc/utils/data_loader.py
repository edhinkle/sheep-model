# Modeled after: https://github.com/NERSC/nersc-dl-multigpu/blob/main/utils/data_loader.py

import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import sys
import subprocess
import os

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

    dataset = ShowerDataset(params, data_location, mode=mode)
    # define a sampler for distributed training using DDP
    sampler = DistributedSampler(dataset, shuffle=train, drop_last=True) if distributed else None
    batch_size = int(params.local_batch_size if train else params.local_valid_batch_size if not test else params.local_test_batch_size)

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=params.num_data_workers,
                            shuffle=(sampler is None and train), # Don't shuffle validation/test data
                            sampler=sampler,
                            collate_fn=shower_collate_fn,
                            drop_last=True,
                            pin_memory=torch.cuda.is_available(), 
                            timeout=120) # timeout to prevent hanging

    return dataloader, sampler

segments_event_data_dtype = np.dtype([
    ('dE', np.float32),  # Energy deposit
    ('x', np.float32),   # X coordinate
    ('y', np.float32),   # Y coordinate
    ('z', np.float32)    # Z coordinate
])

# Dataset collate function for batching
def shower_collate_fn(batch): 
    """Collate function to combine multiple samples into a batch."""
    #coords_list = []
    #features_list = []
    data_list = []
    labels_list = []
    ve_frac_list = []
    mg_frac_list = []
    oob_frac_list = []

    for batch_idx, (data, label, VE_frac, MG_frac, OOB_frac) in enumerate(batch):
        # Set batch index
        batch_data = data.clone()
        #print("Batch data shape:", batch_data.shape)
        batch_data[:, 0] = batch_idx  # Set batch index to the first column

        data_list.append(batch_data)

        labels_list.append(label)
        ve_frac_list.append(VE_frac)
        mg_frac_list.append(MG_frac)
        oob_frac_list.append(OOB_frac)
    
    # Concatenate all coordinates and features into single tensors
    batched_data = torch.cat(data_list, dim=0)
    batched_labels = torch.stack(labels_list, dim=0).reshape(-1, 1) # ESSENTIAL -- ensures output and labels are same shape
    batched_ve_frac = torch.stack(ve_frac_list, dim=0).reshape(-1, 1)
    batched_mg_frac = torch.stack(mg_frac_list, dim=0).reshape(-1, 1)
    batched_oob_frac = torch.stack(oob_frac_list, dim=0).reshape(-1, 1)

    #print("Batched data:", batched_data)

    return batched_data, batched_labels, batched_ve_frac, batched_mg_frac, batched_oob_frac


class ShowerDataset(Dataset):
         
    def __init__(self, params, data_location, mode='train'):

        """
        Args:
            params:                Parameters from yaml file
            data_location:         Path to dataset directory
            mode:                  'train' | 'valid' | 'test' 
        """

        self.mode = mode
       
        self._file_dir = data_location
        # Set num_files for train/validation/test based on mode
        self._num_files_train = params.train_files
        self._num_files_val = params.val_files
        self._num_files_test = params.test_files
        self._set_dataset_file_list()  # Get list of files in dataset directory
        self._set_events_per_file()  # Get number of events per file + file indices

        self._start_pos_range = np.array(params.start_xyz_range)  # Range for sampling random start position
        self._detector_active_regions = np.array(params.detector_active_regions)
        self._RANDOM_SEED = int(params.random_seed)
        self._voxel_size = np.array(params.voxel_size)
        self._min_visible_energy = params.min_visible_energy  # For numerical stability in MinkowskiEngine (Minimum energy target) 
        self._fid_vol_cut = params.fid_vol_cut  # Fiducial volume cut in cm
        self._train_logE = params.train_logE  # Whether to take log of energy for training targets (for better training stability)

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
        self._num_voxels = np.ceil((self._max_xyz - self._min_xyz) / self._voxel_size).astype(int)


        # Set random seeds for reproducibility
        # Use local generators vs. global to avoid interference between workers and encourage reproducibility
        #self._train_rng = np.random.default_rng(self._RANDOM_SEED)
        #np.random.seed(self._RANDOM_SEED)
        #torch.manual_seed(self._RANDOM_SEED)
        self._epoch = 0
        #self._train_rng = np.random.default_rng(self._RANDOM_SEED)
        self._train_rng = None

        # Set torch device
        #self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #print(f"Using device: {self._device}")
        #torch.device(self._device)

    def __len__(self):
        return np.sum(self._events_per_file)

    def __getitem__(self, idx):

        file_idx, event_local_idx = self._decode_idx(idx)  # Decode the global index into file and event indices
        h5_file_name = self._file_list[file_idx]
        with h5py.File(h5_file_name, 'r') as h5_file:  # Open the HDF5 file
            file_events = h5_file['events']  # Access the events dataset
            file_segments = h5_file['segments']
            file_segments_refs = h5_file['segments_ref']

            event = file_events[event_local_idx]
            event_id = int(event['event_id'])
            segments = file_segments[file_segments_refs[event_id]]

            true_KE_initial = float(np.sqrt(np.sum(np.square(event['pxyz_start']))))
            #if true_KE_initial < 20:
            #    print("Initial KE:", true_KE_initial)
            #    print("Total dE:", np.sum(segments['dE']))  # Debugging line to check total dE for the event

            # Same for train/validation/test now
            rng = self._get_deterministic_rng_per_event(event_id)
            vis_segs, ve_frac, mg_frac, oob_frac = self._get_filtered_segments(rng, segments, true_KE_initial, self._min_visible_energy)
        #filtered_segments = transformed_segments[segments_det_mask]
        
        # Voxelize the filtered segments SPARSELY
        coords, features = self._voxelize_sparse(vis_segs)

        # Convert coords, features, labels, to PyTorch tensors
        coords = torch.from_numpy(coords).contiguous()  # Keep as int32 for voxel indices
        features = torch.from_numpy(features).contiguous()  # Already float32
        combined_data = torch.cat([coords.float(), features], dim=1)  # Convert coords to float only for concatenation

        #true_KE_initial_tensor = torch.tensor(true_KE_initial / 1000.0, dtype=torch.float32) # Convert to GeV
        # Instead of converting to GeV, take log of energy to target output range
        true_KE_initial_tensor = torch.tensor(true_KE_initial, dtype=torch.float32) 
        if self._train_logE == True:
            true_KE_initial_tensor = torch.tensor(true_KE_initial, dtype=torch.float32) 
            true_KE_initial_tensor = torch.log(true_KE_initial_tensor)  
        else:
            true_KE_initial_tensor = torch.tensor(true_KE_initial / 1000.0, dtype=torch.float32) # Convert to GeV
        #print("Took log of true KE initial for better training stability. Original KE:", true_KE_initial, "Log KE:", true_KE_initial_tensor.item())
        # Convert energy fractions to torch tensors
        ve_frac_tensor = torch.tensor(ve_frac, dtype=torch.float32)
        mg_frac_tensor = torch.tensor(mg_frac, dtype=torch.float32)
        oob_frac_tensor = torch.tensor(oob_frac, dtype=torch.float32)

        return combined_data, true_KE_initial_tensor, ve_frac_tensor, mg_frac_tensor, oob_frac_tensor
    

    # Method to get file_idx, event_idx pair from global idx
    def _decode_idx(self, idx):
        """Decode a global index into a file index and an event index."""
        file_idx = np.digitize(idx, self._event_total_by_file) - 1  # Find the file index
        #print("File index:", file_idx)  # Debugging line to check file index
        event_idx = idx - self._event_total_by_file[file_idx]  # Find the event index within that file
        return file_idx, event_idx
    
    # Set deterministic rng per event for validation/testing
    def _get_deterministic_rng_per_event(self, event_id: int):
        """Get a deterministic random number generator for a given event and epoch for determinstic training and validation poses."""
        mixed_seed = (self._RANDOM_SEED * 1_000_0003) ^ (int(event_id) * 97) ^ (int(self._epoch) * 1_000_000 + 1337)
        return np.random.default_rng(np.uint64(mixed_seed & 0xFFFFFFFFFFFFFFFF))  # Ensure seed fits in uint64

    # Method to convert random start position and start direction to set of filtered visible depositions
    def _get_filtered_segments(self, rng, segments, true_KE_initial, min_visible_energy=5.0):
        ''' Method to convert random start position and start direction to set of filtered visible depositions
        Inputs:
            - rng: random number generator (different for train/val/test for reproducibility)
            - segments: array of segments with dE, x, y, z 
            - true_KE_initial: initial kinetic energy of the shower (for calculating visible energy fraction)
            - min_visible_energy: minimum visible energy required (in MeV)
        Outputs:
            - transformed segments: array of segments with transformed positions
            - in_any_volume: boolean array indicating whether each segment is within any detector volume
            - start_pos: the sampled start position
            - rotation_matrix: the sampled rotation matrix
        '''
        # Get segment positions in correct format
        segment_positions = np.array([segments['x'], segments['y'], segments['z']]).T # Shape (N, 3) where N is the number of segments
        segment_energy = np.array([segments['dE']]).T  # Shape (N, 1) where N is the number of segments

        # Sample a random start position and rotation matrix until the visible energy fraction is above the minimum threshold
        # This ensures that the sampled shower has enough visible energy depositions in the detector volumes
        # TO-DO: Figure out how to sample start position and rotation matrix such that the visible energy fraction is above the minimum threshold more often on first try
        # Save best try:
        max_VE_under_threshold = 0.
        best_transformed_segments = segment_positions
        best_in_any_volume = np.any(np.all((best_transformed_segments[:, None, :] >= self._min_bounds) &
                           (best_transformed_segments[:, None, :] <= self._max_bounds), axis=2),
                            axis=1)
        try:    
            for _ in range(5000):
            
                start_pos = self._sample_random_start_position(rng)  # Sample a random start position
                rotation_matrix = self._sample_random_rotation_matrix(rng)  # Sample a random rotation matrix
    
                #print("A few segment positions:", segment_positions[:5])  # Debugging line to check segment positions
                #transformed_segments_xyz = segment_positions @ rotation_matrix.T + start_pos # Shape (N, 3) where N is the number of segments
                transformed_segments_xyz = np.einsum('ij, kj->ki', rotation_matrix, segment_positions) + start_pos  # Efficient matrix multiplication and addition
    
                # Check if each segment is within min or max bounds for each dimension
                # Then, check if xyz of segment is within min and max bounds for any volume
                # This is done by checking if the segment is within the bounds of any detector module
                in_any_volume = np.any(
                    np.all((transformed_segments_xyz[:, None, :] >= self._min_bounds) &
                           (transformed_segments_xyz[:, None, :] <= self._max_bounds), axis=2),
                    axis=1
                )
    
                # Check visible energy depositions
                visible_energy = np.sum(segment_energy[in_any_volume])  # Sum the energy depositions of visible segments
            
                #if visible_energy_fraction >= min_visible_energy:
                if visible_energy >= self._min_visible_energy:
                    break
                elif visible_energy > max_VE_under_threshold:
                    max_VE_under_threshold = visible_energy
                    best_transformed_segments = transformed_segments_xyz
                    best_in_any_volume = in_any_volume
                else: continue
        except:
            print(f"Resampling timed out. Best visible energy achieved was {max_VE_under_threshold} and will be used.", flush=True)
            visible_energy = max_VE_under_threshold
            transformed_segments_xyz = best_transformed_segments
            in_any_volume = best_in_any_volume

        # Get visible energy fraction, module gap energy fraction, and uncontained energy fraction
        # visible_energy_fraction is calculated above

        # Debug determinism in sampling
        #print("True KE Initial: ", true_KE_initial) 
        #print("Start position: ", start_pos)
        #print("Rotation matrix: ", rotation_matrix)
        # Module gap energy 
        in_abs_det_bounds = np.all((transformed_segments_xyz[:, :] >= self._min_xyz) &
                                   (transformed_segments_xyz[:, :] <= self._max_xyz), axis=1)
        in_mod_gaps = in_abs_det_bounds & ~in_any_volume
        module_gap_energy = np.sum(segment_energy[in_mod_gaps])

        # Uncontained energy 
        out_of_detector = ~in_abs_det_bounds
        out_of_det_bounds_energy = np.sum(segment_energy[out_of_detector])

        # Calculate energy fractions
        visible_energy_fraction = visible_energy / true_KE_initial  # Calculate the fraction of visible energy compared to the initial kinetic energy
        module_gap_energy_fraction = module_gap_energy / true_KE_initial
        out_of_det_bounds_energy_fraction = out_of_det_bounds_energy / true_KE_initial

        #print("Length of masks:", len(in_abs_det_bounds))
        #print("Sum of in_mod_gaps mask:", np.sum(in_mod_gaps))
        #print("Sum of in_any_volume mask:", np.sum(in_any_volume))
        #print("Sum of oob mask:", np.sum(out_of_detector))
        #print("Sum of mask sums:", np.sum(in_mod_gaps)+np.sum(in_any_volume)+np.sum(out_of_detector))


        #print("Visible Energy Fraction:", visible_energy_fraction)
        #print("Module Gap Energy Fraction:", module_gap_energy_fraction)
        #print("Out of Bounds Energy Fraction:", out_of_det_bounds_energy_fraction)
        #print("Sum of Energy Fractions:", visible_energy_fraction+module_gap_energy_fraction+out_of_det_bounds_energy_fraction)
        #
        #if visible_energy_fraction < min_visible_energy:
        if visible_energy < 0.:
            raise RuntimeError(f"Not enough visible energy ({visible_energy}) for event with initial KE {true_KE_initial}. Resampling and contingency failed.", flush=True)
            
        visible_xyz = transformed_segments_xyz[in_any_volume]  # Get the positions of visible segments
        visible_dE = segment_energy[in_any_volume]  # Get the energy
        visible_segments = np.zeros(visible_xyz.shape[0], dtype=segments_event_data_dtype)  # Create an empty array for visible segments
        visible_segments['dE'] = visible_dE.flatten()  # Fill the energy depositions
        visible_segments['x'] = visible_xyz[:, 0]  # Fill the x coordinate
        visible_segments['y'] = visible_xyz[:, 1]  # Fill the y coordinate
        visible_segments['z'] = visible_xyz[:, 2]  # Fill the z coordinate

        return visible_segments, visible_energy_fraction, module_gap_energy_fraction, out_of_det_bounds_energy_fraction

  
    # Method to get uniformly randomly sampled directions
    def _sample_random_rotation_matrix(self, rng):
        """Generate a random rotation matrix -- rng depends on train/val/test mode for reproducibility"""


        # Step 1: Uniform spherical sampling (use z->theta to reduce oversampling at poles) 
        phi = rng.uniform(0, 2 * np.pi)
        z = rng.uniform(-1, 1)
        theta = np.arccos(z) 

        # Step 2: Get direction vector
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
        #x_start = rng.uniform(self._start_pos_range[0][0], self._start_pos_range[1][0])
        #y_start = rng.uniform(self._start_pos_range[0][1], self._start_pos_range[1][1])
        #z_start = rng.uniform(self._start_pos_range[0][2], self._start_pos_range[1][2])

        # restrict start position to the detector active regions
        num_modules = len(self._detector_active_regions)
        module_idx = rng.integers(0, num_modules)  # Randomly select a detector module
        x_start = rng.uniform(self._detector_active_regions[module_idx, 0, 0]+self._fid_vol_cut, 
                              self._detector_active_regions[module_idx, 1, 0]-self._fid_vol_cut)
        y_start = rng.uniform(self._detector_active_regions[module_idx, 0, 1]+self._fid_vol_cut, 
                              self._detector_active_regions[module_idx, 1, 1]-self._fid_vol_cut)
        z_start = rng.uniform(self._detector_active_regions[module_idx, 0, 2]+self._fid_vol_cut, 
                              self._detector_active_regions[module_idx, 1, 2]-self._fid_vol_cut)

        return np.array([x_start, y_start, z_start])
    
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

        for file in glob.glob(self._file_dir + '*.hdf5'):
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
                events = f['events']
                self._events_per_file.append(len(events))
        self._events_per_file = np.array(self._events_per_file)
        self._event_total_by_file = np.cumsum(self._events_per_file)  # Cumulative sum to get event indices
        self._event_total_by_file = np.insert(self._event_total_by_file, 0, 0)

    def _unique_voxel_indices(self, coords, features):
        """Get unique voxel indices and their corresponding features by combining
        features at the same voxel index."""

        if len(coords) <= 1:
            return coords, features # nothing to combine
        
        # Find unique coordinates and aggregate features
        #  (use np.void for structured array/bytewise comparison vs. elementwise)
        coord_view = coords.view(np.dtype((np.void, coords.dtype.itemsize * coords.shape[1])))
        unique_coords, inverse_indices = np.unique(coord_view, return_inverse=True)

        # Aggregate features at the same voxel index
        unique_features = np.zeros((len(unique_coords), 1), dtype=features.dtype)
        for i in range(len(unique_coords)):
            indices_mask = (inverse_indices == i)
            unique_features[i] = np.sum(features[indices_mask], axis=0)  # Sum features at the same voxel index

        # Convert unique_coords back to original dtype
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

        return self._unique_voxel_indices(coords, features)
    
   