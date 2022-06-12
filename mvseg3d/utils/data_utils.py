import torch
import numpy as np


def load_data_to_gpu(data_dict):
    for key, val in data_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        else:
            if key in ['point_indices', 'point_voxel_ids', 'labels']:
                data_dict[key] = torch.from_numpy(val).long().cuda()
            else:
                data_dict[key] = torch.from_numpy(val).float().cuda()
    return data_dict


def get_shuffled_sub_indices(sub_indices, shuffled_all_indices):
    sub_indices_dic = {}
    for index in sub_indices:
        sub_indices_dic[index] = True

    pos_in_all_indices = []
    shuffled_sub_indices = []
    for i, idx in enumerate(shuffled_all_indices):
        if idx in sub_indices_dic:
            pos_in_all_indices.append(i)
            shuffled_sub_indices.append(idx)
    return pos_in_all_indices, shuffled_sub_indices
