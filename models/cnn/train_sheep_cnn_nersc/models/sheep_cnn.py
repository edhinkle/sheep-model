"""
SHEEP CNN based on SPINE SparseNetEncoder
"""

import torch
import MinkowskiEngine as ME

import sys
sys.path.insert(0, '/global/cfs/cdirs/dune/users/ehinkle/nd_prototypes_ana/sheep-model/models/cnn/spine/')
from spine.model.layer.cnn.encoder import SparseResidualEncoder

def sheep_cnn(params, **kwargs):
    model = SparseResidualEncoder(reps=params.reps, depth=params.depth, filters=params.filters, input_kernel=params.input_kernel, \
                                  coord_conv=params.coord_conv, pool_mode=params.pool_mode, spatial_size=params.spatial_size, \
                                  num_input=params.num_input, feature_size=params.feature_size, allow_bias=params.allow_bias,**kwargs)
    return model


## Pass spatial_size as largest number of voxels -- total is 289x280x292
#sheep_cnn_model = SparseResidualEncoder(reps=2, depth=9, filters=32, \
#                                        input_kernel=7, coord_conv=True, pool_mode="sum",\
#                                        spatial_size=292, num_input=4, feature_size=1, allow_bias=False)