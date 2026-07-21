#!/usr/bin/env python3

import os
import sys
import subprocess


SLURM_NNODES = int(os.environ['SLURM_NNODES'])
SLURM_NTASKS_PER_NODE = int(os.environ['SLURM_NTASKS_PER_NODE'])
SLURM_NODEID = int(os.environ['SLURM_NODEID'])
SLURM_LOCALID = int(os.environ['SLURM_LOCALID']) # the local task ID on the node
GLOBAL_TASK_ID = SLURM_NODEID * SLURM_NTASKS_PER_NODE + SLURM_LOCALID

nfiles = 840
nevents = 50
steps="all"
sample_name = "electron_NDLAr_10MeVto15GeV_1008TEST"
OUTDIR="/pscratch/sd/e/ehinkle/nd_ana/sheep_single_shower/NDLAR_ELECTRON_SAMPLES"
#sample_name = "photon_2x2_10MeVto2GeV_100k"


def main():

    files_per_task = nfiles // SLURM_NTASKS_PER_NODE

    start_idx = GLOBAL_TASK_ID * files_per_task
    end_idx = start_idx + files_per_task

    for idx in range(start_idx, end_idx):

        file_seed = f"{idx:07d}"
        final_hdf5 = os.path.join(
            OUTDIR,
            "HDF5",
            f"{sample_name}.{file_seed}.LARCV2HDF5.hdf5",
        )

        if os.path.exists(final_hdf5):
            print(f"[skip] {final_hdf5}")
            continue

        subprocess.run([f'./run_make_electron_sample.sh', sample_name, OUTDIR, str(nevents), str(idx), steps])
        #subprocess.run([f'./run_make_electron_sample.sh', 'electron_2x2_10MeVto2GeV_100k', str(nevents), str(idx), steps])


if __name__ == '__main__':
    main()
