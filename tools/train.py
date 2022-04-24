import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.optim

from mvseg3d.datasets.waymo_dataset import WaymoDataset
from mvseg3d.models.segmentors.mvf import MVFNet
from mvseg3d.core.metrics import IOUMetric


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
        '--train_batch_size',
        default=4,
        type=int)
    parser.add_argument(
        '--test_batch_size',
        default=8,
        type=int)
    parser.add_argument(
        '--num_workers',
        default=2,
        type=int)
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


def train_one_epoch(train_loader, model, optimizer, lr_scheduler, epoch):
    for step, data_dict in enumerate(train_loader):
        load_data_to_gpu(data_dict)
        out, loss = model(data_dict)
        if step % 5 == 0:
            print('train@epoch:%d, step:%d, loss:%f', (epoch, step, loss.cpu().item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    lr_scheduler.step()


def eval_one_epoch(val_loader, model, iou_metric, epoch):
    model.eval()
    for step, data_dict in enumerate(val_loader):
        load_data_to_gpu(data_dict)
        with torch.no_grad():
            out, loss = model(data_dict)
        if step % 5 == 0:
            print('eval@epoch:%d, step:%d, loss:%f', (epoch, step, loss.cpu().item()))

        pred_labels = torch.argmax(out, dim=1).cpu()
        gt_labels = data_dict['labels']
        iou_metric.add(pred_labels, gt_labels)

    metric_result = iou_metric.get_metric()
    print('Metrics on validation dataset:%s', str(metric_result))


def main():
    args = parse_args()

    # load data
    train_dataset = WaymoDataset(args.data_dir, 'validation')
    train_loader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, pin_memory=True, shuffle=True,
        num_workers=args.num_workers, collate_fn=train_dataset.collate_batch,
        drop_last=False, sampler=None, timeout=0
    )

    val_dataset = WaymoDataset(args.data_dir, 'validation')
    val_loader = DataLoader(
        val_dataset, batch_size=args.test_batch_size, pin_memory=True, shuffle=False,
        num_workers=args.num_workers, collate_fn=val_dataset.collate_batch,
        drop_last=False, sampler=None, timeout=0
    )

    iou_metric = IOUMetric(val_dataset.id2label)

    # define model
    model = MVFNet(train_dataset).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    # train and evaluation
    for epoch in range(args.epochs):
        train_one_epoch(train_loader, model, optimizer, lr_scheduler, epoch)
        eval_one_epoch(val_loader, model, iou_metric, epoch)


if __name__ == '__main__':
    main()
