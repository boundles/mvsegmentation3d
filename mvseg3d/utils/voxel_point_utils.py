import torch

def voxel_to_point(voxel_features, voxel_ids):
    """Gather point's feature from voxel.
    Args:
        voxel_features (num_voxels, C)
        voxel_ids (N)
    Returns:
        point_features: (N, C)
    """
    res_feature_shape = (voxel_ids.shape[0], *voxel_features.shape[1:])
    res = torch.zeros(res_feature_shape, dtype=voxel_features.dtype, device=voxel_features.device)
    voxel_ids_valid = torch.nonzero(voxel_ids != -1).view(-1)
    voxel_features_valid = voxel_features[voxel_ids[voxel_ids_valid]]
    res[voxel_ids_valid] = voxel_features_valid
    return res