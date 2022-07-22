import torch
import numpy as np


def load_data_to_gpu(data_dict):
    for key, val in data_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        else:
            if key in ['point_indices', 'point_voxel_ids', 'voxel_labels', 'labels']:
                data_dict[key] = torch.from_numpy(val).long().cuda()
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
    polar_points = np.concatenate((rho, phi, points[:, 2], points[:, :2], points[:, 3:]), axis=1)
    return polar_points
