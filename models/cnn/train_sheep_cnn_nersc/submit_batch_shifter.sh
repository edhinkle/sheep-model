#!/bin/bash -l
#SBATCH --time=0:20:00
#SBATCH --constraint=gpu
#SBATCH --account=dune
#SBATCH --qos=regular
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --job-name=sheep-dl-500k-weighted-l1-lin-2x2Electrons
#SBATCH --image=deeplearnphysics/larcv2:ub22.04-cuda12.1-pytorch2.4.0-larndsim
#SBATCH --module=cvmfs,gpu,nccl-2.18
#SBATCH --output=shifter_job_log_%j.out

config_file=./configs/2x2_sheep.yaml
config="weighted_l1_lin"
run_num="WeightedL1-LinEnergyScale-500kSample-2x2Electrons-3200bs-1000vbs"
#"WeightedMSE-LinearEnergyScale-50kSample-NDLArTest-HDF5-Test2"

# this is the path to your local env for libs on top of the container
# here we have created a local dir in our ~/.local/perlmutter path
# to mirror what the modules do by default
#env=/global/homes/e/ehinkle/.local/perlmutter/python-3.11

# for DDP
export MASTER_ADDR=$(hostname)

cmd="python train_sheep_multi_gpu.py --yaml_config=$config_file --config=$config --run_num=$run_num"

module load python
set -x
srun -l shifter \
    bash -c "
    set +o posix
    source export_DDP_vars.sh
    $cmd
    " 

# Modeled after: https://github.com/NERSC/nersc-dl-multigpu/blob/main/submit_batch_shifter.sh
