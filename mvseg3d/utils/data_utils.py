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
