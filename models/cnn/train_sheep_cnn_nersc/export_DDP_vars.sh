# exports copied from https://github.com/NERSC/nersc-dl-multigpu/blob/main/export_DDP_vars.sh

# 1) Remove the problematic deps dir that contains an OpenSSL-1.1 libcurl (causes conflict with h5py)
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" \
    | tr ":" "\n" \
    | grep -v "/opt/udiImage/modules/nccl-2.18/deps/lib" \
    | paste -sd:)

# 2) (Optional) Make sure system libs are preferred
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Sanity check: should show only libssl.so.3/libcrypto.so.3
ldd /usr/lib/x86_64-linux-gnu/libcurl.so.4 | egrep "ssl|crypto" || true
python -c 'import h5py; print("h5py OK")'

export RANK=$SLURM_PROCID
export WORLD_RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$SLURM_NTASKS
export MASTER_PORT=29500 # default from torch launcher

# DDP debug flags
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
# Optional, to rule out fabric quirks briefly (slower):
# export NCCL_IB_DISABLE=1