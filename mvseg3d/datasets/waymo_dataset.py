import os
import glob
import numpy as np
from collections import defaultdict

from torch.utils.data import Dataset

from mvseg3d.core import VoxelGenerator
from mvseg3d.datasets.transforms import transforms


class WaymoDataset(Dataset):
    def __init__(self, root, split='training', test_mode=False, use_image_feature=False):
        assert split in ['training', 'validation', 'testing']
        self.root = root
        self.split = split
        self.test_mode = test_mode
        self.use_image_feature = use_image_feature

        if self.test_mode:
            self.filenames = [self.get_filename(path) for path in
                              glob.glob(os.path.join(self.root, split, 'image_feature/0', '*.npy'))]
        else:
            self.filenames = [self.get_filename(path) for path in
                              glob.glob(os.path.join(self.root, split, 'label', '*.npy'))]

        self.voxel_generator = VoxelGenerator(voxel_size=[0.05, 0.05, 0.05],
                                              point_cloud_range=[-75.2, -75.2, -2, 75.2, 75.2, 4],
                                              max_num_points=5,
                                              max_voxels=150000)

        self.grid_size = self.voxel_generator.grid_size
        self.voxel_size = self.voxel_generator.voxel_size
        self.point_cloud_range = self.voxel_generator.point_cloud_range

        self.transforms = transforms.Compose([transforms.RandomGlobalScaling([0.95, 1.05]),
                                              transforms.RandomGlobalRotation([-0.78539816, 0.78539816]),
                                              transforms.PointShuffle()])

    @property
    def dim_point(self):
        return 6

    @property
    def dim_image_feature(self):
        return 66

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

    def get_point_image_features(self, num_points, filename):
        # assemble all camera features
        image_features = dict()
        for i in range(5):
            image_feature_file = os.path.join(self.root, self.split, 'image_feature', str(i),  filename + '.npy')
            image_feature = np.load(image_feature_file, allow_pickle=True).item()
            image_features.update(image_feature)

        # get point image features
        point_image_features = np.zeros((num_points, self.dim_image_feature), dtype=np.float32)
        for k in image_features:
            point_image_features[k] = image_features[k]
        return point_image_features

    def get_lidar(self, filename):
        lidar_file = os.path.join(self.root, self.split, 'lidar', filename + '.npy')
        # (N, 15): [x, y, z, range, intensity, elongation, 6-dim camera project, range col, row and index]
        # when test mode, otherwise (N, 12) without range col, row and index
        lidar_points = np.load(lidar_file)

        # normalize intensity
        lidar_points[:, 4] = np.tanh(lidar_points[:, 4])
        return lidar_points[:, :self.dim_point]

    def get_label(self, filename):
        label_file = os.path.join(self.root, self.split, 'label', filename + '.npy')
        semantic_labels = np.load(label_file)[:, 1]  # (N, 1)

        # convert unlabeled to ignored label (0 to 255)
        semantic_labels -= 1
        semantic_labels[semantic_labels == -1] = 255
        return semantic_labels

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]

        points = self.get_lidar(filename)
        input_dict = {
            'points': points[:, :self.dim_point]
        }

        if self.use_image_feature:
            point_image_features = self.get_point_image_features(points.shape[0], filename)
            input_dict['point_image_features'] = point_image_features

        if not self.test_mode:
            labels = self.get_label(filename)
            input_dict['labels'] = labels

        if self.test_mode:
            input_dict['points_ri'] = points[:, -3:]
            input_dict['frame_id'] = filename

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
        if self.split == 'training':
            data_dict = self.transforms(data_dict)

        voxels, coords, num_points_per_voxel, point_voxel_ids = self.voxel_generator.generate(data_dict['points'])
        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coords
        data_dict['voxel_num_points'] = num_points_per_voxel
        data_dict['point_voxel_ids'] = point_voxel_ids

        return data_dict

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)

        ret = {}
        voxel_nums = None
        for key, val in data_dict.items():
            if key in ['voxel_coords']:
                coors = []
                voxel_nums = []
                for i, coor in enumerate(val):
                    coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                    coors.append(coor_pad)
                    voxel_nums.append(coor.shape[0])
                ret[key] = np.concatenate(coors, axis=0)
            elif key in ['points', 'points_ri', 'point_image_features', 'labels', 'voxels', 'voxel_num_points']:
                ret[key] = np.concatenate(val, axis=0)
            elif key in ['frame_id']:
                ret[key] = val

        offset = 0
        if voxel_nums and 'point_voxel_ids' in data_dict:
            point_voxel_ids_list = data_dict['point_voxel_ids']
            for i, point_voxel_ids in enumerate(point_voxel_ids_list):
                point_voxel_ids[point_voxel_ids != -1] += offset
                offset += voxel_nums[i]
            ret['point_voxel_ids'] = np.concatenate(point_voxel_ids_list, axis=0)

        batch_size = len(batch_list)
        ret['batch_size'] = batch_size
        return ret


if __name__ == '__main__':
    dataset = WaymoDataset('/nfs/dataset-dtai-common/waymo_open_dataset_v_1_3_0', 'validation')
    for step, sample in enumerate(dataset):
        print(step, sample['points'].shape, sample['labels'].shape)
