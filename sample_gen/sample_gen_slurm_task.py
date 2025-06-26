#!/usr/bin/env python3

import os
import sys


SLURM_NNODES = int(os.environ['SLURM_NNODES'])
SLURM_NTASKS_PER_NODE = int(os.environ['SLURM_NTASKS_PER_NODE'])
SLURM_NODEID = int(os.environ['SLURM_NODEID'])
SLURM_LOCALID = int(os.environ['SLURM_LOCALID']) # the local task ID on the node
GLOBAL_TASK_ID = SLURM_NODEID * SLURM_NTASKS_PER_NODE + SLURM_LOCALID

nfiles = 20
nevents = 50


def main():

    files_per_task = nfiles // SLURM_NTASKS_PER_NODE

    start_idx = GLOBAL_TASK_ID * files_per_task
    end_idx = start_idx + files_per_task

    for idx in range(start_idx, end_idx):
        os.system(f'./run_make_sample.sh test_electron {nevents} {idx}')


if __name__ == '__main__':
    main()
