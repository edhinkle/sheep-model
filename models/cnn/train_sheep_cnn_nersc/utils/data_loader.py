# Modeled after: https://github.com/NERSC/nersc-dl-multigpu/blob/main/utils/data_loader.py

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import sys
import subprocess
import os
import json
import math
import torch.distributed as dist
import traceback
import time

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

    if train or not 'valid':
        # Only freeze eval poses for validation/testing
        freeze_eval_poses = False
        dataset = ShowerDataset(params, data_location, mode=mode, freeze_eval_poses=freeze_eval_poses)
        # define a sampler for distributed training using DDP
        sampler = DistributedSampler(dataset, shuffle=train, drop_last=True) if distributed else None
        batch_size = int(params.local_batch_size if train else params.local_valid_batch_size)

        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=params.num_data_workers,
                                shuffle=(sampler is None and train), # Don't shuffle validation/test data
                                sampler=sampler,
                                collate_fn=shower_collate_fn,
                                drop_last=True,
                                pin_memory=torch.cuda.is_available(), 
                                timeout=120) # timeout to prevent hanging
    
    else:
        # Used cached validation dataset for validation
        freeze_eval_poses = True
        raw_validation = ShowerDataset(params, data_location, mode=mode, freeze_eval_poses=freeze_eval_poses)

        # Check if validation cache exists, if not create it
        cache_file_exists = raw_validation._check_val_cache()
        if cache_file_exists:
            print(f"Validation cache file exists at {raw_validation.val_cache_file_path_final} ...", flush=True)
        else:
            print(f"Unlogged error in checking/creating validation cache file ...", flush=True)

        # Get cached validation dataset
        cached_val_dataset = CachedValDataset(raw_validation.val_cache_file_path_final)

        # Iterate per event, use batch size of 1 and no collate_fn
        sampler = None # order is deterministic
        if distributed==True and dist.get_rank() == 0:
            dataloader = DataLoader(cached_val_dataset,
                          batch_size=1, 
                          num_workers=params.num_data_workers, # can single thread for h5 access
                          shuffle=False, 
                          sampler=sampler,
                          collate_fn=None,
                          drop_last=False,
                          pin_memory=torch.cuda.is_available(),
                          timeout=120) # timeout to prevent hanging
        else:
            dataloader = DataLoader([],
                          batch_size=1, 
                          num_workers=params.num_data_workers, # can single thread for h5 access
                          shuffle=False, 
                          sampler=sampler,
                          collate_fn=None,
                          drop_last=False,
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

    return batched_data, batched_labels, batched_ve_frac, batched_mg_frac, batched_oob_frac


class ShowerDataset(Dataset):
         
    def __init__(self, params, data_location, mode='train', freeze_eval_poses=True):

        """
        Args:
            params:                Parameters from yaml file
            data_location:         Path to dataset directory
            mode:                  'train' | 'valid' | 'test' 
            freeze_eval_poses:     If True, use fixed poses for validation/testing (deterministic evaluation)
        """

        self.mode = mode
        self.freeze_eval_poses = freeze_eval_poses
        self.eval_poses_per_event = params.eval_poses_per_event
        self.val_cache_path = params.val_cache_path if hasattr(params, 'val_cache_path') else None
        self.val_cache_filename = f"val_cache_K{params.eval_poses_per_event}_seed{int(params.random_seed)}_minVE{int(params.min_visible_energy)}MeV"
        self.val_cache_file_tmp = self.val_cache_filename+".tmp.h5"
        self.val_cache_file_final = self.val_cache_filename+".h5"
        self.val_cache_file_path_tmp = os.path.join(self.val_cache_path, self.val_cache_file_tmp) if hasattr(params, 'val_cache_path') else None
        self.val_cache_file_path_final = os.path.join(self.val_cache_path, self.val_cache_file_final) if hasattr(params, 'val_cache_path') else None

        self._file_dir = data_location
        self._set_dataset_file_list()  # Get list of files in dataset directory
        self._set_events_per_file()  # Get number of events per file + file indices

        self._start_pos_range = np.array(params.start_xyz_range)  # Range for sampling random start position
        self._detector_active_regions = np.array(params.detector_active_regions)
        self._RANDOM_SEED = int(params.random_seed)
        self._voxel_size = np.array(params.voxel_size)
        self._min_visible_energy = params.min_visible_energy  # For numerical stability in MinkowskiEngine (Minimum energy target) 
        self._fid_vol_cut = params.fid_vol_cut  # Fiducial volume cut in cm

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
        self._train_rng = np.random.default_rng(self._RANDOM_SEED)
        #np.random.seed(self._RANDOM_SEED)
        #torch.manual_seed(self._RANDOM_SEED)

        # Set torch device
        #self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #print(f"Using device: {self._device}")
        #torch.device(self._device)

        # In-memory cache so we only compute a valid pose once per (event_id, pose_idx) -- used for eval only
        self._eval_pose_cache = {}  # key: (event_id, pose_idx) -> (visible_segments, ve_frac, mg_frac, oob_frac)


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

            if self.mode == 'train':
                rng = self._train_rng
                # Get filtered segments for training
                vis_segs, ve_frac, mg_frac, oob_frac = self._get_filtered_segments(rng, segments, true_KE_initial)
            else:
                # For validation/testing, use deterministic poses
                if self.freeze_eval_poses:
                    key = (event_id, 0) # pose_idx = 0 until expanding to >1 poses per event
                    if key not in self._eval_pose_cache:
                        rng = self._get_deterministic_rng_per_event(event_id, pose_idx=0)
                        self._eval_pose_cache[key] = self._get_filtered_segments(rng, segments, true_KE_initial)
                    # Retrieve from cache
                    vis_segs, ve_frac, mg_frac, oob_frac = self._eval_pose_cache[key]
                else:
                    # Non-deterministic eval poses (different each epoch -- NOT ideal)
                    rng = self._train_rng
                    vis_segs, ve_frac, mg_frac, oob_frac = self._get_filtered_segments(rng, segments, true_KE_initial)
        #filtered_segments = transformed_segments[segments_det_mask]
        
        # Voxelize the filtered segments SPARSELY
        coords, features = self._voxelize_sparse(vis_segs)

        # Convert coords, features, labels, to PyTorch tensors
        coords = torch.from_numpy(coords).contiguous()  # Keep as int32 for voxel indices
        features = torch.from_numpy(features).contiguous()  # Already float32
        combined_data = torch.cat([coords.float(), features], dim=1)  # Convert coords to float only for concatenation

        true_KE_initial_tensor = torch.tensor(true_KE_initial / 1000.0, dtype=torch.float32) # Convert to GeV

        # Convert energy fractions to torch tensors
        ve_frac_tensor = torch.tensor(ve_frac, dtype=torch.float32)
        mg_frac_tensor = torch.tensor(mg_frac, dtype=torch.float32)
        oob_frac_tensor = torch.tensor(oob_frac, dtype=torch.float32)

        return combined_data, true_KE_initial_tensor, ve_frac_tensor, mg_frac_tensor, oob_frac_tensor
    
    # Method to build evaluation cache in temp location, then rename as permanent (help w/ no timeout on other ranks for DDP)
    def _build_eval_cache_full(self):

        # Check that cache directory exists
        os.makedirs(self.val_cache_path, exist_ok=True)

        # Check whether final file exists
        if os.path.exists(self.val_cache_file_path_final):
            return True
        
        # Check whether there is a temp file from an old attempt
        if os.path.exists(self.val_cache_file_path_tmp):
            try:
                os.remove(self.val_cache_file_path_tmp)
            except OSError:
                pass

        # Try building cache
        try:
            print(f"[rank 0] Precomputing validation cache in {self.val_cache_file_path_tmp} first...", flush=True)
            self._precompute_eval_poses_cache()
            os.replace(self.val_cache_file_path_tmp, self.val_cache_file_path_final)
            print(f"[rank 0] Validation cache ready in {self.val_cache_file_path_final}", flush=True)
        except Exception:
            # Clean up temp too
            try:
                if os.path.exists(self.val_cache_file_path_tmp):
                    os.remove(self.val_cache_file_path_tmp)
            except OSError:
                pass
            print(f"[rank 0] cache build FAILED\n{traceback.format_exc()}", flush=True)
            raise

        return True
        

    # Method to check if validation poses are cached
    def _check_val_cache(self):

        if self.mode != 'valid' or not self.freeze_eval_poses:
            return None  # No caching needed for training or non-frozen eval poses
        
        # Check validation caching for multi-gpu
        rank = 0
        world_size = 1
        ddp = False

        try: 
            ddp = dist.is_initialized()
            if ddp:
                rank = dist.get_rank()
                world_size = dist.get_world_size()
        except Exception:
            pass

        finished = False
        if rank == 0:
            if not os.path.isfile(self.val_cache_file_path_final):
                finished = self._build_eval_cache_full()
            else:
                print(f"[rank 0] Validation cache file {self.val_cache_file_path_final} already exists, skipping precomputation.", flush=True)
        else:
            if "SLURM_TIMELIMIT" in os.environ: # variable will be in minutes -- convert and only take 90% of that time
                finished = self._wait_for_eval_cache_ready(max_wait_time=int(os.environ["SLURM_TIMELIMIT"] * 60 * 0.9))
            else:
                finished = self._wait_for_eval_cache_ready()

        if ddp:
            dist.barrier()  # wait for rank 0 to finish precomputing cache

        return finished

    # Method to get file_idx, event_idx pair from global idx
    def _decode_idx(self, idx):
        """Decode a global index into a file index and an event index."""
        file_idx = np.digitize(idx, self._event_total_by_file) - 1  # Find the file index
        #print("File index:", file_idx)  # Debugging line to check file index
        event_idx = idx - self._event_total_by_file[file_idx]  # Find the event index within that file
        return file_idx, event_idx
    
    # Set deterministic rng per event for validation/testing
    def _get_deterministic_rng_per_event(self, event_id: int, pose_idx: int=0):
        """Get a deterministic random number generator for a given event and pose index for cross-epoch stability"""
        mixed_seed = (self._RANDOM_SEED * 1_000_0003) ^ (int(event_id) * 97) ^ (pose_idx * 1_000_000 + 1337)
        return np.random.default_rng(np.uint64(mixed_seed & 0xFFFFFFFFFFFFFFFF))  # Ensure seed fits in uint64

    # Method to convert random start position and start direction to set of filtered visible depositions
    def _get_filtered_segments(self, rng, segments, true_KE_initial):
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

    # Method to get multiple poses per event for eval
    def  _get_multi_pose(self, event_id, segments, true_KE_initial):
        """Get multiple poses per event for evaluation"""
        poses = []
        for pose_idx in range(self.eval_poses_per_event):
            rng = self._get_deterministic_rng_per_event(event_id, pose_idx)
            vis_segs, ve_frac, mg_frac, oob_frac = self._get_filtered_segments(rng, segments, true_KE_initial)
            coords, features = self._voxelize_sparse(vis_segs)
            coords = torch.from_numpy(coords).contiguous()  # Keep as int32 for voxel indices
            features = torch.from_numpy(features).contiguous()  # Already float32
            combined_data = torch.cat([coords.float(), features], dim=1)  # Convert coords to float only for concatenation
            poses.append((combined_data, ve_frac, mg_frac, oob_frac))
        return poses
    
    # Method to precompute eval poses and save to cache
    def _precompute_eval_poses_cache(self):
        """Precompute eval poses and save to cache directory file"""
        os.makedirs(self.val_cache_path, exist_ok=True)

        # Write to temp file_path
        cache_file_path = self.val_cache_file_path_tmp
        # Open an output HDF5 file to store the cached poses
        with h5py.File(cache_file_path, 'w') as h5_cache_file:

            event_ids = []
            labels = []
            ve_all = [] # shape [N, eval_poses_per_event]
            mg_all = []
            oob_all = []

            # variable length datatypes for coords (int32) and features (float32)
            vlen_i32 = h5py.vlen_dtype(np.dtype('int32'))
            vlen_f32 = h5py.vlen_dtype(np.dtype('float32'))

            # Create datasets to store concatenated poses per event + offsets to slice per pose
            dataset_length = self.__len__()
            coords_list = h5_cache_file.create_dataset('coords_list', (dataset_length,), dtype=vlen_i32)
            features_list = h5_cache_file.create_dataset('features_list', (dataset_length,), dtype=vlen_f32)
            pose_offsets = h5_cache_file.create_dataset('pose_offsets', (dataset_length, self.eval_poses_per_event + 1), dtype='int64')

            # Look through validation sample once
            for idx in range(dataset_length):
                file_idx, event_local_idx = self._decode_idx(idx)
                with h5py.File(self._file_list[file_idx], 'r') as val_file:

                    event = val_file['events'][event_local_idx]
                    event_id = int(event['event_id'])
                    segments = val_file['segments'][val_file['segments_ref'][event_id]]
                    true_KE_initial = float(np.sqrt(np.sum(np.square(event['pxyz_start']))))

                # Get multiple poses for the event
                pose_vec = []
                ve_vec, mg_vec, oob_vec = [], [], []
                offsets = [0]
                for pose_idx in range(self.eval_poses_per_event):
                    rng = self._get_deterministic_rng_per_event(event_id, pose_idx)
                    vis_segs, ve_frac, mg_frac, oob_frac = self._get_filtered_segments(rng, segments, true_KE_initial)
                    coords, features = self._voxelize_sparse(vis_segs)
                    pose_vec.append((coords, features))
                    ve_vec.append(ve_frac); mg_vec.append(mg_frac); oob_vec.append(oob_frac)
                    offsets.append(offsets[-1] + coords.shape[0])

                # Store concatenated coordinates and features
                coords_concat = np.concatenate([c for (c, _) in pose_vec], axis=0)
                features_concat = np.concatenate([f for (_, f) in pose_vec], axis=0)

                # Store in HDF5 datasets
                event_ids.append(event_id)
                labels.append(true_KE_initial) 
                ve_all.append(ve_vec)
                mg_all.append(mg_vec)
                oob_all.append(oob_vec)

                coords_list[idx] = coords_concat.reshape(-1).astype(np.int32)
                features_list[idx] = features_concat.reshape(-1).astype(np.float32)
                pose_offsets[idx, :] = np.array(offsets, dtype=np.int64)

            # Write fixed-size arrays
            h5_cache_file.create_dataset('event_ids',     data=np.array(event_ids, dtype='int64'), compression="gzip")
            h5_cache_file.create_dataset('labels',        data=np.array(labels,    dtype='float32'), compression="gzip")
            h5_cache_file.create_dataset('ve_fractions',  data=np.array(ve_all,    dtype='float32'), compression="gzip")
            h5_cache_file.create_dataset('mg_fractions',  data=np.array(mg_all,    dtype='float32'), compression="gzip")
            h5_cache_file.create_dataset('oob_fractions', data=np.array(oob_all,   dtype='float32'), compression="gzip")

            # Add metadata
            h5_cache_file.attrs['eval_poses_per_event'] = self.eval_poses_per_event
            h5_cache_file.attrs['random_seed'] = self._RANDOM_SEED
            h5_cache_file.attrs['version'] = 'v1.0'


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

        for file in glob.glob(self._file_dir + '*.hdf5'):
            self._file_list.append(file)

        if len(self._file_list) == 0:
            raise ValueError("No files found in dataset directory: {}".format(self._file_dir)) 
        
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
    
    # Method for ranks > 0 to wait for validation cache readiness before moving on
    def _wait_for_eval_cache_ready(self, max_wait_time=3600*10, idle_timeout=900, checking_frequency=2, stability_time=4):
        """
            max_wait_time:      HARD CAP maximum time until timeout from waiting for validation cache file to be built [s]
            idle_timeout:       maximum wait time if temp file size is not changing [s]
            checking_frequency: how often nodes check for file/file size [s]
            stability_time:     how long to wait for final file size to stabilize [s]
        """

        start_time = time.monotonic()
        last_size = -1
        last_change = start_time

        # Wait for file to exist
        while True:

            # Get current time
            now = time.monotonic()
            
            # Check if final file exists
            if os.path.exists(self.val_cache_file_path_final):
                # Wait for file size to stop changing (precaution for multi-GPU/complex system)
                size = os.path.getsize(self.val_cache_file_path_final)
                if size == last_size:
                    if now - last_change >= stability_time:
                        return True
                else:
                    last_size = size
                    last_change = now
            # Check for temp file existing and changing
            else:
                if os.path.exists(self.val_cache_file_path_tmp):
                    size = os.path.getsize(self.val_cache_file_path_tmp)
                    if size != last_size:
                        last_size = size
                        last_change = now

                if now - last_change > idle_timeout:
                    raise TimeoutError(f"No progress on temporary validation cache file for {idle_timeout}s;"
                                       f"Temp file exists={os.path.exists(self.val_cache_file_path_tmp)}")
                
            # Hard cap timeout
            if now - start_time > max_wait_time:
                raise TimeoutError(f"Timed out waiting for {self.val_cache_file_path_final} to appear.")
            
            time.sleep(checking_frequency)
                




# Class for cached validation dataset
class CachedValDataset(Dataset):

    def __init__(self, cache_file_path):

        self.cache_file_path = cache_file_path
        self.h5_file = h5py.File(self.cache_file_path, 'r')
        self.event_ids = self.h5_file['event_ids'][:]            # [N]
        self.labels = self.h5_file['labels'][:]                  # [N]
        self.ve_fractions = self.h5_file['ve_fractions'][:]      # [N, eval_poses_per_event]
        self.mg_fractions = self.h5_file['mg_fractions'][:]      # [N, eval_poses_per_event]
        self.oob_fractions = self.h5_file['oob_fractions'][:]    # [N, eval_poses_per_event]
        self.coords_list_vlen = self.h5_file['coords_list']      # variable length dataset
        self.features_list_vlen = self.h5_file['features_list']  # variable length dataset
        self.pose_offsets = self.h5_file['pose_offsets']         # [N, eval_poses_per_event + 1]
        self.eval_poses_per_event = self.h5_file.attrs['eval_poses_per_event']

    def __len__(self):
        return len(self.event_ids)
    
    def __getitem__(self, idx):
        # Get concatenated coords and features for the event
        coords_concat = self.coords_list_vlen[idx][:]
        features_concat = self.features_list_vlen[idx][:]
        offsets = self.pose_offsets[idx]
        eval_poses_per_event = self.eval_poses_per_event

        # Give list of poses for the event (val loop iterates and averages)
        poses = []
        for k in range(eval_poses_per_event):
            start, end = int(offsets[k]), int(offsets[k+1])
            coords = coords_concat[start*4 : end*4].reshape(-1, 4).astype(np.float32)
            features = features_concat[start : end].reshape(-1, 1).astype(np.float32)
            combined_data = np.concatenate([coords, features], axis=1)
            poses.append(torch.from_numpy(combined_data).contiguous())

        label = torch.tensor(self.labels[idx] / 1000., dtype=torch.float32).reshape(-1, 1) # Convert to GeV in same place as for ShowerDataset dataloader
        ve_fractions = torch.from_numpy(self.ve_fractions[idx]).to(torch.float32)
        mg_fractions = torch.from_numpy(self.mg_fractions[idx]).to(torch.float32)
        oob_fractions = torch.from_numpy(self.oob_fractions[idx]).to(torch.float32)

        return poses, label, ve_fractions, mg_fractions, oob_fractions