import json
import os
import glob
import numpy as np

from torch.utils.data import Dataset


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
        semantic_labels = np.load(label_file)[:, 1] # (N, 1)
        return semantic_labels

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, index):
        sample_idx = self.file_ids[index]
        points = self.get_lidar(sample_idx)
        labels = self.get_label(sample_idx)

        data_dict = {
            'points': points,
            'labels': labels
        }
        return data_dict


if __name__ == '__main__':
    dataset = WaymoDataset('/nfs/volume-807-2/waymo_open_dataset_v_1_3_0', 'validation')
    for step, sample in enumerate(dataset):
        print(step, sample['points'].shape, sample['labels'].shape)