#!/bin/bash -l
#SBATCH --time=4:00:00
#SBATCH --constraint=gpu
#SBATCH --account=dune
#SBATCH --qos=regular
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --job-name=sheep-dl
#SBATCH --image=deeplearnphysics/larcv2:ub22.04-cuda12.1-pytorch2.4.0-larndsim
#SBATCH --module=cvmfs,gpu,nccl-2.18
#SBATCH --output=shifter_job_log_%j.out

config_file=./configs/medium_modular.yaml
results_dir=./paper_outputs

config="mse_log"
run_num="MSE-LogEnergyScale"

# for DDP
export MASTER_ADDR=$(hostname)

cmd="python train_sheep_multi_gpu.py --yaml_config=$config_file --config=$config --results_dir=$results_dir --run_num=$run_num"

module load python
set -x
srun --cpu-bind=cores -l shifter \
    bash -c "
    set +o posix
    source export_DDP_vars.sh
    $cmd
    " 

# Modeled after: https://github.com/NERSC/nersc-dl-multigpu/blob/main/submit_batch_shifter.sh
