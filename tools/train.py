import numpy as np

import torch
from torch.utils.data import DataLoader

from mvseg3d.datasets.waymo_dataset import WaymoDataset
from mvseg3d.models.segmentors.mvf import MVFNet


def load_data_to_gpu(data_dict):
    for key, val in data_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        else:
            data_dict[key] = torch.from_numpy(val).float().cuda()
    return data_dict


def main():
    # load data
    dataset = WaymoDataset('/nfs/volume-807-2/waymo_open_dataset_v_1_3_0', 'validation')
    dataloader = DataLoader(
        dataset, batch_size=4, pin_memory=True, num_workers=2,
        shuffle=True, collate_fn=dataset.collate_batch,
        drop_last=False, sampler=None, timeout=0
    )

    model = MVFNet(dataset).cuda()
    for step, data_dict in enumerate(dataloader, 1):
        load_data_to_gpu(data_dict)
        y = model(data_dict)
        print(step, type(y))


if __name__ == '__main__':
    main()
