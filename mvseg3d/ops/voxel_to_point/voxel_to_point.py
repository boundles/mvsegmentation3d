import torch


def voxel_to_point(feats, coords):
    """Gather point's feature from voxel.
    Args:
        feats (num_voxels, C)
        coords (N)
    Returns:
        point_features: (N, C)
    """
    res_feature_shape = (coords.shape[0], feats.shape[-1])
    res = torch.zeros(res_feature_shape, dtype=feats.dtype, device=feats.device)
    coords_valid = torch.nonzero(coords != -1).view(-1)
    features_valid = feats[coords[coords_valid]]
    res[coords_valid] = features_valid
    return res