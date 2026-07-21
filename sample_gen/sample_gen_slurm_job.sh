#!/usr/bin/env bash

#SBATCH --account=dune
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --time=6:30:00
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1

srun ./sample_gen_slurm_task.py

