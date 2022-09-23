import torch

from mvseg3d.ops import knn_query


def query_and_group(nsample, xyz, new_xyz, feat, idx, offset, new_offset, use_xyz=True):
    """
    input: xyz: (n, 3), new_xyz: (m, 3), feat: (n, c), idx: (m, nsample), offset: (b), new_offset: (b)
    output: new_feat: (m, c+3, nsample), grouped_idx: (m, nsample)
    """
    assert xyz.is_contiguous() and new_xyz.is_contiguous() and feat.is_contiguous()
    if new_xyz is None:
        new_xyz = xyz
    if idx is None:
        idx, _ = knn_query(nsample, xyz, new_xyz, offset, new_offset)  # (m, nsample)

    n, m, c = xyz.shape[0], new_xyz.shape[0], feat.shape[1]
    grouped_xyz = xyz[idx.view(-1).long(), :].view(m, nsample, 3)  # (m, nsample, 3)
    grouped_xyz -= new_xyz.unsqueeze(1)  # (m, nsample, 3)
    grouped_feat = feat[idx.view(-1).long(), :].view(m, nsample, c)  # (m, nsample, c)

    if use_xyz:
        return torch.cat((grouped_xyz, grouped_feat), -1)  # (m, nsample, 3+c)
    else:
        return grouped_feat
