import torch

from mvseg3d.models.layers import SparseWindowPartitionLayer, WindowAttention


if __name__ == '__main__':
    drop_info = {
        0: {'max_tokens': 30, 'drop_range': (0, 30)},
        1: {'max_tokens': 60, 'drop_range': (30, 60)},
        2: {'max_tokens': 100, 'drop_range': (60, 100000)}
    }
    window_shape = (10, 10, 10)
    sparse_shape = (400, 400, 20)
    normalize_pos = False
    pos_temperature = 10000
    window_partition = SparseWindowPartitionLayer(drop_info, window_shape, sparse_shape)

    voxel_features = torch.randn((4, 24)).cuda()
    voxel_coords = torch.zeros((4, 4)).cuda()
    voxel_info = window_partition(voxel_features, voxel_coords)

    window_attention = WindowAttention(24, 2, 0.5).cuda()
    result = window_attention(voxel_info['voxel_features'], voxel_info['pos_dict_shift0'],
                              voxel_info['flat2win_inds_shift0'], voxel_info['key_mask_shift0'])
    print(result)


