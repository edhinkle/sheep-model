#!/usr/bin/env bash

#SBATCH --account=dune_g
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --time=4:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32

srun ./sample_gen_slurm_task.py
