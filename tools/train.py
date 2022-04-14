import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.optim

from mvseg3d.datasets.waymo_dataset import WaymoDataset
from mvseg3d.models.segmentors.mvf import MVFNet


def load_data_to_gpu(data_dict):
    for key, val in data_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        else:
            if key in ['point_voxel_ids', 'labels']:
                data_dict[key] = torch.from_numpy(val).long().cuda()
            else:
                data_dict[key] = torch.from_numpy(val).float().cuda()
    return data_dict


def parse_args():
    parser = argparse.ArgumentParser(description='Train a 3d segmentor')
    parser.add_argument(
        '--data_dir',
        type=str,
        help='the data directory')
    parser.add_argument(
        '--epochs',
        default=10,
        type=int)
    parser.add_argument(
        '--lr',
        default=0.01,
        type=float)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # load data
    dataset = WaymoDataset(args.data_dir, 'validation')
    data_loader = DataLoader(
        dataset, batch_size=4, pin_memory=True, num_workers=2,
        shuffle=True, collate_fn=dataset.collate_batch,
        drop_last=False, sampler=None, timeout=0
    )

    # define model
    model = MVFNet(dataset).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    # train
    for epoch in range(args.epochs):
        for step, data_dict in enumerate(data_loader, 1):
            load_data_to_gpu(data_dict)
            out, loss = model(data_dict)
            print('epoch:', epoch, ', step:', step, ', loss:', loss)

            loss.backward()
            optimizer.zero_grad()
            optimizer.step()
        scheduler.step()


if __name__ == '__main__':
    main()
