import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.optim

from mvseg3d.datasets.waymo_dataset import WaymoDataset
from mvseg3d.models.segmentors.mvf import MVFNet
from mvseg3d.core.metrics import IOUMetric
from mvseg3d.utils.logging import get_logger

logger = get_logger("mvseg3d")


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
    parser.add_argument(
        '--log_interval',
        default=5,
        type=int
    )
    parser.add_argument(
        '--eval_interval',
        default=1000,
        type=int
    )

    args = parser.parse_args()
    return args


def evaluate(data_loaders, model, id2label, args):
    cur_iter = 0
    total_iter = args.epochs * len(data_loaders['val'])
    model.eval()
    iou_metric = IOUMetric(id2label)
    for epoch in range(args.epochs):
        for step, data_dict in enumerate(data_loaders['val']):
            load_data_to_gpu(data_dict)
            with torch.no_grad():
                out, loss = model(data_dict)
            cur_iter += 1
            if cur_iter % args.log_interval == 0:
                logger.info(
                    'Iter [%d/%d] loss: %f' % (cur_iter, total_iter, loss.cpu().item()))

            pred_labels = torch.argmax(out, dim=1).cpu()
            gt_labels = data_dict['labels'].cpu()
            iou_metric.add(pred_labels, gt_labels)

    metric_result = iou_metric.get_metric()
    logger.info('Metrics on validation dataset: %s' % str(metric_result))


def train_segmentor(data_loaders, id2label, model, optimizer, lr_scheduler, args):
    cur_iter = 0
    total_iter = args.epochs * len(data_loaders['train'])
    model.train()
    for epoch in range(args.epochs):
        for step, data_dict in enumerate(data_loaders['train']):
            load_data_to_gpu(data_dict)
            out, loss = model(data_dict)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cur_iter += 1
            if cur_iter % args.log_interval == 0:
                logger.info(
                    'Iter [%d/%d] lr: %f, loss: %f' % (cur_iter, total_iter, lr_scheduler.get_last_lr()[0], loss.cpu().item()))

            if cur_iter % args.eval_interval == 0:
                logger.info('Evaluate on epoch: %d' % epoch)
                evaluate(data_loaders, model, id2label, args)
        lr_scheduler.step()


def main():
    args = parse_args()

    # load data
    train_dataset = WaymoDataset(args.data_dir, 'training')
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

    data_loaders = {'train': train_loader, 'val': val_loader}

    logger.info('Loaded %d train samples, batch size: %d' % (len(train_dataset), args.train_batch_size))
    logger.info('Loaded %d validation samples, batch size: %d' % (len(val_dataset), args.test_batch_size))

    # define model
    model = MVFNet(train_dataset).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    # train and evaluation
    train_segmentor(data_loaders, val_dataset.id2label, model, optimizer, lr_scheduler, args)


if __name__ == '__main__':
    main()
