#!/usr/bin/env python3
"""
Converts LArCV dataset to sharded HDF5 files, adapted from the edep-sim
ROOT->HDF5 conversion script (single_shower_convert_edepsim_roottoh5.py)
"""

import numpy as np
import h5py
from tqdm import tqdm
from pathlib import Path
import fire
import sys
sys.path.insert(0, '/global/cfs/cdirs/dune/users/ehinkle/nd_prototypes_ana/sheep-model/models/cnn/spine/src/spine')
import spine
from spine.io.dataset.larcv import LArCVDataset


def init_hdf5_shard(output_file: str, chunk_voxels: int = 65536):
    with h5py.File(output_file, "w", libver="latest") as f:
        f.create_dataset("ke_initial", (0,), dtype=np.float32, maxshape=(None,))
        f.create_dataset("start_points", (0, 3), dtype=np.float32, maxshape=(None, 3))
        f.create_dataset(
            "voxels_flat",
            shape=(0, 4),
            dtype=np.float32,
            maxshape=(None, 4),
            chunks=(chunk_voxels, 4),   # now data-driven
            compression="lzf",
        )
        f.create_dataset(
            "voxels_offsets",
            shape=(1,),
            dtype=np.int64,
            maxshape=(None,),
        )
        f["voxels_offsets"][0] = 0
        f.attrs["n_events"] = 0


def update_hdf5_shard(output_file: str, ke_batch: np.ndarray, start_points_batch: np.ndarray, voxels_batch: list):
    """
    Append a batch of events to an existing HDF5 shard.

    Parameters
    ----------
    ke_batch    : float32 (batch_size,)       — KE per event
    start_points_batch: float32 (batch_size, 3) — start points per event
    voxels_batch: list of float32 (N_i, 4)   — voxel arrays per event
    """
    if len(ke_batch) == 0:
        return

    voxels_flat = np.concatenate(voxels_batch, axis=0)  # (total_voxels_in_batch, 4)

    with h5py.File(output_file, "a") as f:
        # ── ke_initial ────────────────────────────────────────────────────────
        n_existing = len(f["ke_initial"])
        f["ke_initial"].resize((n_existing + len(ke_batch),))
        f["ke_initial"][n_existing:] = ke_batch

        # ── start_points ──────────────────────────────────────────────────────
        n_start_existing = len(f["start_points"])
        f["start_points"].resize((n_start_existing + len(start_points_batch), 3))
        f["start_points"][n_start_existing:] = start_points_batch

        # ── voxels_flat ───────────────────────────────────────────────────────
        n_vox_existing = f["voxels_flat"].shape[0]
        f["voxels_flat"].resize((n_vox_existing + len(voxels_flat), 4))
        f["voxels_flat"][n_vox_existing:] = voxels_flat

        # ── voxels_offsets ────────────────────────────────────────────────────
        # Build new offset entries relative to the current flat buffer end
        last_offset = int(f["voxels_offsets"][-1])
        new_offsets = np.zeros(len(voxels_batch), dtype=np.int64)
        for i, arr in enumerate(voxels_batch):
            new_offsets[i] = last_offset + len(arr)
            last_offset = new_offsets[i]

        n_off_existing = len(f["voxels_offsets"])
        f["voxels_offsets"].resize((n_off_existing + len(new_offsets),))
        f["voxels_offsets"][n_off_existing:] = new_offsets

        f.attrs["n_events"] = n_existing + len(ke_batch)


def convert(
    root_file: str,
    output_file: str,
    larcv_config: dict[str, dict[str, str]] = {"input_data": {
                                                "parser": "sparse3d",
                                                "sparse_event": "sparse3d_pcluster",
                                                },
                                                "particles": {
                                                    "parser": "particle",
                                                    "particle_event": "particle_pcluster",
                                                    "sparse_event": "sparse3d_pcluster",
                                                },
                                                "meta": {
                                                    "parser": "meta",
                                                    "sparse_event": "sparse3d_pcluster",
                                                },
                                               },
    write_every: int = 100,
    chunk_voxels: int = 65536,     # set from your median voxel count
):

    print(f"{root_file} → {output_file}")

    # One LArCVDataset per ROOT file
    ds = LArCVDataset(file_keys=[str(root_file)], schema=larcv_config, dtype="float32")
    n = len(ds)

    init_hdf5_shard(output_file, chunk_voxels=chunk_voxels)

    ke_accum     = []
    start_points_accum = []
    voxels_accum = []

    for i in tqdm(range(n)):
        data = ds[i]

        ke = float(data['particles'][0].p)

        if data['particles'][0].pdg_code == 22:
            original_particle_id = data['particles'][0].id
            # Find the first child of the photon (if any)
            child_particles = [p for p in data['particles'] if p.parent_id == original_particle_id]
            child_particles = child_particles[1:]  # Skip the photon itself
            if len(child_particles) < 1:
                print(f"Warning: Photon with ID {original_particle_id} has no children. Using photon's start position.")
                start_point = np.asarray([data['particles'][0].x, data['particles'][0].y, data['particles'][0].z])
            else:
                first_child_idx = np.argmin([p.t for p in child_particles])  # Skip the photon itself
                start_point = np.asarray(data['meta'].to_cm(child_particles[first_child_idx].position, center=False), dtype=np.float32)
                print("Start point for photon event (ID {}): {}".format(original_particle_id, start_point))
        else:
            start_point = np.asarray([data['particles'][0].x, data['particles'][0].y, data['particles'][0].z])

        energy_per_voxel = np.asarray(
            data['input_data'].features, dtype=np.float32
        ).reshape(-1, 1)

        positions_cm = np.asarray(
            data['meta'].to_cm(data['input_data'].coords, center=True),
            dtype=np.float32
        )

        voxels = np.concatenate([positions_cm, energy_per_voxel], axis=1)

        ke_accum.append(ke)
        start_points_accum.append(start_point)
        voxels_accum.append(voxels)

        if len(ke_accum) >= write_every or i == n - 1:
            update_hdf5_shard(
                output_file,
                np.array(ke_accum, dtype=np.float32),
                np.array(start_points_accum, dtype=np.float32),
                voxels_accum
            )
            ke_accum     = []
            start_points_accum = []
            voxels_accum = []

    print(f"\nDone. Events written to {output_file}/")


if __name__ == "__main__":
    fire.Fire(convert)
