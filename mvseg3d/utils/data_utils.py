import torch
import numpy as np


def load_data_to_gpu(data_dict):
    for key, val in data_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        else:
            if key in ['voxel_labels', 'labels']:
                data_dict[key] = torch.from_numpy(val).long().cuda()
            elif key in ['point_indices', 'point_voxel_ids']:
                data_dict[key] = torch.from_numpy(val).int().cuda()
            else:
                data_dict[key] = torch.from_numpy(val).float().cuda()
    return data_dict


def get_sub_indices_pos(sub_indices, all_indices):
    sub_indices_dic = {}
    for i, idx in enumerate(sub_indices):
        sub_indices_dic[idx] = i

    pos_in_all_indices = []
    pos_in_sub_indices = []
    for i, idx in enumerate(all_indices):
        if idx in sub_indices_dic:
            pos_in_all_indices.append(i)
            pos_in_sub_indices.append(sub_indices_dic[idx])
    return pos_in_all_indices, pos_in_sub_indices


# transformation between Cartesian coordinates and polar coordinates
def cart2polar(points):
    rho = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    phi = np.arctan2(points[:, 1], points[:, 0])
    return np.stack((rho, phi, points[:, 2]), axis=1)


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids
