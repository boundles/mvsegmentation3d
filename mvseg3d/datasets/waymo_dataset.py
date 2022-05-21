import json
import os
import glob
import numpy as np
from collections import defaultdict

import torch
from torch.utils.data import Dataset

from mvseg3d.core import VoxelGenerator


class WaymoDataset(Dataset):
    def __init__(self, root, split='training', test_mode=False):
        assert split in ['training', 'validation', 'test']
        self.root = root
        self.split = split
        self.test_mode = test_mode

        if self.test_mode:
            self.filenames = [self.get_filename(path) for path in
                              glob.glob(os.path.join(self.root, split, 'lidar', '*.npy'))]
        else:
            self.filenames = [self.get_filename(path) for path in
                              glob.glob(os.path.join(self.root, split, 'label', '*.npy'))]

        self.voxel_generator = VoxelGenerator(voxel_size=[0.1, 0.1, 0.15],
                                              point_cloud_range=[-75.2, -75.2, -2, 75.2, 75.2, 4],
                                              max_num_points=5,
                                              max_voxels=150000)

        self.grid_size = self.voxel_generator.grid_size
        self.voxel_size = self.voxel_generator.voxel_size
        self.point_cloud_range = self.voxel_generator.point_cloud_range

    @property
    def point_dim(self):
        return 6

    @property
    def num_classes(self):
        return 22

    @property
    def id2label(self):
        labels = ['Car', 'Truck', 'Bus', 'Other Vehicle', 'MotorCyclist', 'Bicyclist', 'Pedestrian', 'Sign',
                  'Traffic Light', 'Pole', 'Construction Cone', 'Bicycle', 'MotorCycle', 'Building', 'Vegetation',
                  'Tree Trunk', 'Curb', 'Road', 'Lane Marker', 'Other Ground', 'Walkable', 'SideWalk']
        id2label = dict()
        for i, label in enumerate(labels):
            id2label[i] = label
        return id2label

    @property
    def class_weight(self):
        return [2.5573495292786705, 4.674679353043209, 4.999718438900531, 5.563321749633249, 10.73673169853078,
                8.227823688837296, 4.962283296561223, 5.237239400142882, 7.307259288590015, 4.610857239545126,
                7.607454941285682, 9.15210312179631, 8.753426840231054, 1.1511133803065023, 1.63258363570334,
                3.8735526105804743, 4.443068509630746, 1.6108377340079234, 5.1972959658694355, 5.3112725890845445,
                2.626942890303363, 2.958990264327497]

    @staticmethod
    def get_filename(path):
        return os.path.splitext(os.path.basename(path))[0]

    def get_lidar(self, filename):
        lidar_file = os.path.join(self.root, self.split, 'lidar', filename + '.npy')
        lidar_points = np.load(lidar_file)  # (N, 12): [x, y, z, range, intensity, elongation, camera project]

        # normalize intensity
        lidar_points[:, 4] = np.tanh(lidar_points[:, 4])
        return lidar_points[:, :self.point_dim]

    def get_label(self, filename):
        label_file = os.path.join(self.root, self.split, 'label', filename + '.npy')
        semantic_labels = np.load(label_file)[:, 1]  # (N, 1)

        # convert unlabeled to ignored label (0 to 255)
        semantic_labels -= 1
        semantic_labels[semantic_labels == -1] = 255
        return semantic_labels

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, index):
        filename = self.filenames[index]

        points = self.get_lidar(filename)
        input_dict = {
            'points': points
        }

        if not self.test_mode:
            labels = self.get_label(filename)
            input_dict['labels'] = labels

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
                voxel_num_points: optional (num_voxels)
                point_voxel_ids: optional, (N)
        """
        voxels, coords, num_points_per_voxel, point_voxel_ids = self.voxel_generator.generate(data_dict['points'])
        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coords
        data_dict['voxel_num_points'] = num_points_per_voxel
        data_dict['point_voxel_ids'] = point_voxel_ids

        return data_dict

    @staticmethod
    def load_data_to_gpu(batch_dict):
        for key, val in batch_dict.items():
            if not isinstance(val, np.ndarray):
                continue
            else:
                batch_dict[key] = torch.from_numpy(val).float().cuda()

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)

        ret = {}
        voxel_coords_list = None
        point_voxel_ids_list = None
        for key, val in data_dict.items():
            if key == 'voxel_coords':
                voxel_coords_list = val
                coors = []
                for i, coor in enumerate(voxel_coords_list):
                    coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                    coors.append(coor_pad)
                ret[key] = np.concatenate(coors, axis=0)
            elif key == 'point_voxel_ids':
                point_voxel_ids_list = val
            elif key in ['points', 'labels', 'voxels', 'voxel_num_points']:
                ret[key] = np.concatenate(val, axis=0)

        offset = 0
        if voxel_coords_list and point_voxel_ids_list:
            for i, point_voxel_ids in enumerate(point_voxel_ids_list):
                point_voxel_ids[point_voxel_ids != -1] += offset
                offset += voxel_coords_list[i].shape[0]
            ret['point_voxel_ids'] = np.concatenate(point_voxel_ids_list, axis=0)

        batch_size = len(batch_list)
        ret['batch_size'] = batch_size
        return ret


if __name__ == '__main__':
    dataset = WaymoDataset('/nfs/volume-807-2/waymo_open_dataset_v_1_3_0', 'validation')
    for step, sample in enumerate(dataset):
        print(step, sample['points'].shape, sample['labels'].shape)
