import json
import os
import glob
import numpy as np
from collections import defaultdict

from torch.utils.data import Dataset
from mvseg3d.ops import VoxelGenerator


class WaymoDataset(Dataset):
    def __init__(self, root, split='train', filter_nlz_points=False):
        assert split in ['train', 'validation', 'test']
        self.root = root
        self.split = split
        self.filter_nlz_points = filter_nlz_points

        self.file_ids = [os.path.basename(path).replace('.npy', '') \
                         for path in glob.glob(os.path.join(self.root, split, 'label', '*.npy'))]

    def get_lidar(self, sample_idx):
        lidar_file = os.path.join(self.root, self.split, 'lidar', sample_idx + '.npy')
        lidar_points = np.load(lidar_file)  # (N, 6): [x, y, z, intensity, elongation, timestamp]

        lidar_points[:, 3] = np.tanh(lidar_points[:, 3])
        return lidar_points

    def get_label(self, sample_idx):
        label_file = os.path.join(self.root, self.split, 'label', sample_idx + '.npy')
        semantic_labels = np.load(label_file)[:, 1]  # (N, 1)
        return semantic_labels

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, index):
        sample_idx = self.file_ids[index]

        points = self.get_lidar(sample_idx)
        labels = self.get_label(sample_idx)

        input_dict = {
            'points': points,
            'labels': labels
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

    def prepare_data(self, data_dict):
        """
        Args:
            data_dict:
                points: optional, (N, ndim)
                labels: optional, (N)
        Returns:
            data_dict:
                points: (N, ndim)
                labels: optional, (N)
                voxels: optional (num_voxels, max_points, ndim)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
        """
        voxels, coords, num_points_per_voxel = VoxelGenerator.generate(data_dict['points'])
        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coords
        data_dict['voxel_num_points'] = num_points_per_voxel

        return data_dict

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)

        ret = {}
        batch_size = len(batch_list)
        for key, val in data_dict.items():
            try:
                if key in ['voxels', 'voxel_num_points']:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['points', 'voxel_coords', 'labels']:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError
        ret['batch_size'] = batch_size
        return ret


if __name__ == '__main__':
    dataset = WaymoDataset('/nfs/volume-807-2/waymo_open_dataset_v_1_3_0', 'validation')
    for step, sample in enumerate(dataset):
        print(step, sample['points'].shape, sample['labels'].shape)
