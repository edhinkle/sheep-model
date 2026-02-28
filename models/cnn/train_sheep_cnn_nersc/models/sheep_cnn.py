"""
SHEEP CNN based on SPINE SparseNetEncoder
"""

import torch
import torch.distributed as dist
import MinkowskiEngine as ME

import sys
sys.path.insert(0, '/global/cfs/cdirs/dune/users/ehinkle/nd_prototypes_ana/sheep-model/models/cnn/spine/')
from spine.model.layer.cnn.encoder import SparseResidualEncoder

def sheep_cnn(params, **kwargs):
    model = SparseResidualEncoder(reps=params.reps, depth=params.depth, filters=params.filters, input_kernel=params.input_kernel, \
                                  coord_conv=params.coord_conv, pool_mode=params.pool_mode, spatial_size=params.spatial_size, \
                                  num_input=params.num_input, feature_size=params.feature_size, allow_bias=params.allow_bias,**kwargs)
    
    for m in model.modules():
        if isinstance(m, ME.MinkowskiBatchNorm):
            m.momentum = params.bn_momentum
        if isinstance(m, ME.MinkowskiSyncBatchNorm):
            m.momentum = params.bn_momentum
    
    for name, module in model.named_modules():
        if isinstance(module, ME.MinkowskiBatchNorm):
            module.register_forward_hook(bn_hook(name))
        if isinstance(module, ME.MinkowskiSyncBatchNorm):
            module.register_forward_hook(bn_hook(name))

    # Debug test
    #for name, m in model.named_modules():
    #    if isinstance(m, ME.MinkowskiBatchNorm):
    #        print(f"BatchNorm Layer: {name}, Momentum: {m.momentum}")

    return model

# Hook for getting batch norm batch statistics
batch_norm_stats = {}

def bn_hook(name):
    def hook(module, input, output):

        # Get input tensor to BN layer
        x = input[0]
        coords = x.C
        feats = x.F
        batch_ids = coords[:, 0]  # Assuming batch ID is in the first column of coordinates

        # debug batches
        #print("Num points:", feats.shape[0])
        #print("Num batches:", batch_ids.unique())
        #print("Feature dim:", feats.shape[1])

        # global batch stats
        #batch_mean = feats.mean(dim=0)
        #batch_var = feats.var(dim=0, correction=0)
        batch_mean, batch_var = sync_stats(feats)

        # per-batch statistics
        per_batch_stats = {}

        #for b in batch_ids.unique():
        #    batch_id_mask = batch_ids == b
        #    batch_feats = feats[batch_id_mask]
#
        #    per_batch_stats[int(b)] = {
        #        'batch_mean': batch_feats.mean(dim=0).detach().cpu().numpy(),
        #        'batch_var': batch_feats.var(dim=0, correction=0).detach().cpu().numpy(),
        #        'n_points': batch_feats.shape[0]
        #    }
#

        batch_norm_stats[name] = {
            'global_batch_mean': batch_mean.detach().cpu().numpy(), 
            'global_batch_var': batch_var.detach().cpu().numpy(),
            'running_mean': module.bn.running_mean.detach().cpu().numpy(),
            'running_var': module.bn.running_var.detach().cpu().numpy(),
            'momentum': module.bn.momentum
        }
        #print(f"BatchNorm Layer: {name}, \
        #      Global Batch Mean: {batch_mean[:5].detach().cpu().numpy()}, \
        #      Running Mean: {module.bn.running_mean[:5].detach().cpu().numpy()}, \
        #      Global Batch Var: {batch_var[:5].detach().cpu().numpy()}, \
        #      Running Var: {module.bn.running_var[:5].detach().cpu().numpy()}")
              # Per-batch stats: { {b: {'mean': per_batch_stats[b]['batch_mean'][:5], 'var': per_batch_stats[b]['batch_var'][:5], 'n_points': per_batch_stats[b]['n_points']} for b in per_batch_stats} }")
    return hook

# Add method to calculate mean and variance synchronized across all processes
def sync_stats(feats):

    sum_ = feats.sum(dim=0)
    sq_sum_ = (feats ** 2).sum(dim=0)
    count = torch.tensor(feats.shape[0], device=feats.device)

    dist.all_reduce(sum_)
    dist.all_reduce(sq_sum_)
    dist.all_reduce(count)

    mean = sum_ / count
    var = (sq_sum_ / count) - mean **2
    return mean, var
## Pass spatial_size as largest number of voxels -- total is 289x280x292
#sheep_cnn_model = SparseResidualEncoder(reps=2, depth=9, filters=32, \
#                                        input_kernel=7, coord_conv=True, pool_mode="sum",\
#                                        spatial_size=292, num_input=4, feature_size=1, allow_bias=False)