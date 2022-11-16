import torch

from mvseg3d.models.layers import SparseWindowPartitionLayer


if __name__ == '__main__':
    drop_info = {
        0: {'max_tokens': 30, 'drop_range': (0, 30)},
        1: {'max_tokens': 60, 'drop_range': (30, 60)},
        2: {'max_tokens': 100, 'drop_range': (60, 100000)}
    }
    window_shape = (10, 10, 10)
    sparse_shape = (400, 400, 20)
    normalize_pos = False,
    pos_temperature = 10000
    window_partition = SparseWindowPartitionLayer(drop_info, window_shape, sparse_shape)

    voxel_features = torch.randn((4, 2)).cuda()
    voxel_coords = torch.zeros((4, 4)).cuda()
    result = window_partition(voxel_features, voxel_coords)