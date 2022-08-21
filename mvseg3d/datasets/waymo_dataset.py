import os
import glob

import numpy as np
from collections import defaultdict

from torch.utils.data import Dataset

from mvseg3d.core import VoxelGenerator
from mvseg3d.datasets.transforms import transforms
from mvseg3d.utils.geometry import cart2polar


class WaymoDataset(Dataset):
    def __init__(self, cfg, root, split='training', test_mode=False):
        assert split in ['training', 'validation', 'testing']
        self.cfg = cfg
        self.root = root
        self.split = split
        self.test_mode = test_mode

        if self.test_mode:
            self.filenames = self.get_filenames('image_feature')
        else:
            self.filenames = self.get_filenames('label')

        self.file_idx_to_name_map = dict()
        self.lidar_filenames = self.get_filenames('lidar')
        for filename in self.lidar_filenames:
            file_idx, frame_idx, timestamp = self.parse_info_from_filename(filename)
            self.file_idx_to_name_map[(file_idx, frame_idx)] = filename

        if self.split == 'training':
            mode = 'train'
        else:
            mode = 'test'
        self.voxel_generator = VoxelGenerator(voxel_size=cfg.DATASET.VOXEL_SIZE,
                                              point_cloud_range=cfg.DATASET.POINT_CLOUD_RANGE,
                                              max_num_points=cfg.DATASET.MAX_NUM_POINTS,
                                              max_voxels=cfg.DATASET.MAX_VOXELS[mode])

        self.grid_size = self.voxel_generator.grid_size
        self.voxel_size = self.voxel_generator.voxel_size
        self.point_cloud_range = self.voxel_generator.point_cloud_range

        self.transforms = transforms.Compose([transforms.RandomFlip(),
                                              transforms.RandomGlobalScaling(cfg.DATASET.AUG_SCALE_RANGE),
                                              transforms.RandomGlobalRotation(cfg.DATASET.AUG_ROT_RANGE),
                                              transforms.PointShuffle(),
                                              transforms.PointSample(cfg.DATASET.AUG_SAMPLE_RATIO, cfg.DATASET.AUG_SAMPLE_RANGE),
                                              transforms.RandomDropPointsColor(cfg.DATASET.AUG_DROP_RATIO)])

    @property
    def dim_point(self):
        return self.cfg.DATASET.DIM_POINT

    @property
    def use_cylinder(self):
        return self.cfg.DATASET.USE_CYLINDER

    @property
    def num_classes(self):
        return self.cfg.DATASET.NUM_CLASSES

    @property
    def class_names(self):
        class_names = self.cfg.DATASET.CLASS_NAMES
        return class_names

    @property
    def class_weight(self):
        return self.cfg.DATASET.CLASS_WEIGHT

    @property
    def use_image_feature(self):
        return self.cfg.DATASET.USE_IMAGE_FEATURE

    @property
    def dim_image_feature(self):
        return self.cfg.DATASET.DIM_IMAGE_FEATURE

    @property
    def ignore_index(self):
        return self.cfg.DATASET.IGNORE_INDEX

    def get_filenames(self, dir_name):
        return [os.path.splitext(os.path.basename(path))[0] for path in
                glob.glob(os.path.join(self.root, self.split, dir_name, '*.npy'))]

    def get_lidar_path(self, filename):
        lidar_file = os.path.join(self.root, self.split, 'lidar', filename + '.npy')
        return lidar_file

    def get_image_feature_path(self, filename):
        image_feature_file = os.path.join(self.root, self.split, 'image_feature', filename + '.npy')
        return image_feature_file

    def get_pose_path(self, filename):
        pose_file = os.path.join(self.root, self.split, 'pose', filename + '.txt')
        return pose_file

    def get_label_path(self, filename):
        label_file = os.path.join(self.root, self.split, 'label', filename + '.npy')
        return label_file

    @staticmethod
    def parse_info_from_filename(filename):
        splits = filename.split('-')
        file_idx = splits[0]
        timestamp = np.int64(splits[1])
        frame_idx = int(splits[2])
        return file_idx, frame_idx, timestamp

    def load_pose(self, filename):
        pose_file = self.get_pose_path(filename)
        sensor2local_matrix = np.loadtxt(pose_file)
        return sensor2local_matrix

    def load_image_features(self, num_points, filename):
        # load image feature
        image_feature_file = self.get_image_feature_path(filename)
        image_feature = np.load(image_feature_file, allow_pickle=True).item()

        # assemble point image features
        point_image_features = np.zeros((num_points, self.dim_image_feature), dtype=np.float32)
        for k in image_feature:
            point_image_features[k] = image_feature[k]
        return point_image_features

    def load_points(self, filename):
        lidar_file = self.get_lidar_path(filename)
        # (N, 15): [x, y, z, range, intensity, elongation, 6-dim camera project, range col, row and index]
        # when test mode, otherwise (N, 12) without range col, row and index
        lidar_points = np.load(lidar_file)

        # set range value to be zero
        lidar_points[:, 3] = 0
        # normalize intensity
        lidar_points[:, 4] = np.tanh(lidar_points[:, 4])
        return lidar_points

    def load_points_from_multi_sweeps(self, filename, num_sweeps=3, max_num_sweeps=5, pad_empty_sweeps=False):
        # current frame
        file_idx, frame_idx, timestamp = self.parse_info_from_filename(filename)
        points = self.load_points(filename)
        points[:, 3] = 0
        point_indices = np.arange(points.shape[0])
        ts = timestamp / 1e6
        transform_matrix = self.load_pose(filename)

        # history sweep filenames
        history_sweep_filenames = []
        for i in range(0, max_num_sweeps - 1):
            sweep_frame_idx = frame_idx - i - 1
            if sweep_frame_idx >= 0:
                sweep_filename = self.file_idx_to_name_map[(file_idx, sweep_frame_idx)]
                history_sweep_filenames.append(sweep_filename)

        history_num_sweeps = num_sweeps - 1
        sweep_points_list = [points]
        if pad_empty_sweeps and len(history_sweep_filenames) == 0:
            for i in range(history_num_sweeps):
                sweep_points_list.append(points)
        else:
            if len(history_sweep_filenames) <= history_num_sweeps:
                choices = np.arange(len(history_sweep_filenames))
            elif self.test_mode:
                choices = np.arange(history_num_sweeps)
            else:
                choices = np.random.choice(
                    len(history_sweep_filenames), history_num_sweeps, replace=False)

            for idx in choices:
                sweep_filename = history_sweep_filenames[idx]
                points_sweep = self.load_points(sweep_filename)
                timestamp = self.parse_info_from_filename(sweep_filename)[-1]
                sweep_ts = timestamp / 1e6
                sweep_transform_matrix = self.load_pose(sweep_filename)
                sensor2lidar = np.linalg.inv(transform_matrix) @ sweep_transform_matrix
                sensor2lidar_rotation = sensor2lidar[0:3, 0:3]
                sensor2lidar_translation = sensor2lidar[0:3, 3]
                points_sweep[:, :3] = points_sweep[:, :3] @ sensor2lidar_rotation.T
                points_sweep[:, :3] += sensor2lidar_translation
                points_sweep[:, 3] = ts - sweep_ts
                sweep_points_list.append(points_sweep)

        points = np.concatenate(sweep_points_list, axis=0)
        return point_indices, points


    def load_label(self, filename):
        label_file = self.get_label_path(filename)
        semantic_labels = np.load(label_file)[:, 1]  # (N, 1)

        # convert unlabeled to ignored label (0 to 255)
        semantic_labels -= 1
        semantic_labels[semantic_labels == -1] = 255
        return semantic_labels

    def prepare_voxel_labels(self, data_dict):
        assert self.ignore_index == 255
        label_size = 256
        voxel_label_counter = dict()
        voxels = data_dict.get('voxels', None)
        point_voxel_ids = data_dict.get('point_voxel_ids', None)
        labels = data_dict.get('labels', None)
        if voxels is not None and point_voxel_ids is not None and labels is not None:
            for i in range(point_voxel_ids.shape[0]):
                voxel_id = point_voxel_ids[i]
                label = labels[i]
                if voxel_id != -1:
                    if voxel_id not in voxel_label_counter:
                        counter = np.zeros((label_size,), dtype=np.uint16)
                        counter[label] += 1
                        voxel_label_counter[voxel_id] = counter
                    else:
                        counter = voxel_label_counter[voxel_id]
                        counter[label] += 1
                        voxel_label_counter[voxel_id] = counter

            voxel_labels = np.ones(voxels.shape[0], dtype=np.uint8) * self.ignore_index
            for voxel_id in voxel_label_counter:
                counter = voxel_label_counter[voxel_id]
                voxel_labels[voxel_id] = np.argmax(counter)

            data_dict['voxel_labels'] = voxel_labels

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
        if self.split == 'training' and self.cfg.DATASET.AUG_DATA:
            data_dict = self.transforms(data_dict)

        if self.cfg.DATASET.USE_CYLINDER:
            points = data_dict['points']
            polar_points = cart2polar(points)
            data_dict['points'] = np.concatenate((polar_points, points[:, :2], points[:, 3:]), axis=1)

        voxels, coords, num_points_per_voxel, point_voxel_ids = self.voxel_generator.generate(data_dict['points'])
        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coords
        data_dict['voxel_num_points'] = num_points_per_voxel
        data_dict['point_voxel_ids'] = point_voxel_ids

        point_indices = data_dict.get('point_indices', None)
        if point_indices is not None:
            data_dict['points'] = data_dict['points'][point_indices]
            data_dict['point_voxel_ids'] = data_dict['point_voxel_ids'][point_indices]

        self.prepare_voxel_labels(data_dict)

        return data_dict

    def __getitem__(self, index):
        filename = self.filenames[index]

        input_dict = {
            'filename': filename
        }

        if self.cfg.DATASET.USE_MULTI_SWEEPS:
            point_indices, points = self.load_points_from_multi_sweeps(filename, self.cfg.DATASET.NUM_SWEEPS,
                                                                       self.cfg.DATASET.MAX_NUM_SWEEPS)
            input_dict['points'] = points[:, :self.dim_point]
            input_dict['point_indices'] = point_indices
        else:
            points = self.load_points(filename)
            input_dict['points'] = points[:, :self.dim_point]

        if self.cfg.DATASET.USE_IMAGE_FEATURE:
            point_image_features = self.load_image_features(points.shape[0], filename)
            input_dict['point_image_features'] = point_image_features

        if self.test_mode:
            point_indices = input_dict.get('point_indices', None)
            if point_indices is not None:
                input_dict['points_ri'] = points[point_indices][-3:]
            else:
                input_dict['points_ri'] = points[-3:]
        else:
            labels = self.load_label(filename)
            input_dict['labels'] = labels

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)

        ret = {}
        for key, val in data_dict.items():
            if key in ['points', 'voxel_coords']:
                coors = []
                for i, coor in enumerate(val):
                    coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                    coors.append(coor_pad)
                ret[key] = np.concatenate(coors, axis=0)
            elif key in ['points_ri', 'point_image_features', 'labels', 'voxels', 'voxel_labels']:
                ret[key] = np.concatenate(val, axis=0)
            elif key in ['filename']:
                ret[key] = val

        voxel_offset = 0
        if 'point_voxel_ids' in data_dict and 'voxel_coords' in data_dict:
            point_voxel_ids_list = data_dict['point_voxel_ids']
            voxel_coords_list = data_dict['voxel_coords']
            for i, point_voxel_ids in enumerate(point_voxel_ids_list):
                point_voxel_ids[point_voxel_ids != -1] += voxel_offset
                voxel_offset += voxel_coords_list[i].shape[0]
            ret['point_voxel_ids'] = np.concatenate(point_voxel_ids_list, axis=0)

        batch_size = len(batch_list)
        ret['batch_size'] = batch_size
        return ret

    def __len__(self):
        return len(self.filenames)

if __name__ == '__main__':
    from mvseg3d.utils.config import cfg

    dataset = WaymoDataset(cfg, '/nfs/dataset-dtai-common/waymo_open_dataset_v_1_3_2', 'validation')
    for step, sample in enumerate(dataset):
        print(step, sample['points'].shape, sample['labels'].shape)
