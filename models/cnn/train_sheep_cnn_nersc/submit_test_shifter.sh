#!/bin/bash -l
#SBATCH --time=0:05:00
#SBATCH --constraint=gpu
#SBATCH --account=dune
#SBATCH --qos=regular
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --job-name=sheep-dl-100k-logE-huber-TEST
#SBATCH --image=deeplearnphysics/larcv2:ub22.04-cuda12.1-pytorch2.4.0-larndsim
#SBATCH --module=cvmfs,gpu,nccl-2.18
#SBATCH --output=shifter_job_log_%j.out

yaml_config=./configs/default_sheep.yaml
config="test"
train_config="default"
results_dir=./outputs
run_num="ddp-shifter-MSELOSS-LinearEnergyScale-100kSamples-BUGFIX"
ckpt_file="ckpt_epoch_87_iters_2200_best.tar"

# this is the path to your local env for libs on top of the container
# here we have created a local dir in our ~/.local/perlmutter path
# to mirror what the modules do by default
#env=/global/homes/e/ehinkle/.local/perlmutter/python-3.11

# for DDP
export MASTER_ADDR=$(hostname)

cmd="python test_sheep.py --yaml_config=$yaml_config --config=$config --train_config=$train_config --results_dir=$results_dir --run_num=$run_num --checkpoint_file=$ckpt_file"

set -x
srun -l shifter \
    bash -c "
    set +o posix
    source export_DDP_vars.sh
    $cmd
    " 

# Modeled after: https://github.com/NERSC/nersc-dl-multigpu/blob/main/submit_batch_shifter.sh
