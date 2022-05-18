import os
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
        '--save_dir',
        type=str,
        help='the saved directory')
    parser.add_argument(
        '--batch_size',
        default=4,
        type=int)
    parser.add_argument(
        '--num_gpus',
        default=1,
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
        '--log_iter_interval',
        default=5,
        type=int
    )
    parser.add_argument(
        '--eval_epoch_interval',
        default=1,
        type=int
    )
    parser.add_argument(
        '--auto_resume',
        action='store_true',
        default=True
    )
    args = parser.parse_args()

    # calculate the batch size
    batch_size = args.num_gpus * args.batch_size
    logger.info(f'Training with {args.num_gpus} GPU(s) with {args.batch_size} '
                f'samples per GPU. The total batch size is {batch_size}.')

    if batch_size != args.batch_size:
        # scale LR with
        # [linear scaling rule](https://arxiv.org/abs/1706.02677)
        scaled_lr = (batch_size / args.batch_size) * args.lr
        logger.info('LR has been automatically scaled '
                    f'from {args.lr} to {scaled_lr}')
        args.lr = scaled_lr
        args.batch_size = batch_size
    else:
        logger.info('The batch size match the '
                    f'base batch size: {args.batch_size}, '
                    f'will not scaling the LR ({args.lr}).')

    return args

def save_checkpoint(epoch, model, optimizer, lr_scheduler, save_dir):
    logger.info('Save checkpoint at epoch %d' % epoch)
    checkpoint = {
        "model": model.state_dict(),
        'optimizer': optimizer.state_dict(),
        "epoch": epoch,
        'lr_scheduler': lr_scheduler.state_dict()
    }

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    torch.save(checkpoint, os.path.join(save_dir, 'epoch_%s.pth' % str(epoch)))
    torch.save(checkpoint, os.path.join(save_dir, 'latest.pth'))

def evaluate(data_loader, model, id2label, args):
    iou_metric = IOUMetric(id2label)
    model.eval()
    for step, data_dict in enumerate(data_loader, 1):
        load_data_to_gpu(data_dict)
        with torch.no_grad():
            out, loss = model(data_dict)
        if step % args.log_iter_interval == 0:
            logger.info(
                'Iter [%d/%d] loss: %f' % (step, len(data_loader), loss.cpu().item()))

        pred_labels = torch.argmax(out, dim=1).cpu()
        gt_labels = data_dict['labels'].cpu()
        iou_metric.add(pred_labels, gt_labels)

    metric_result = iou_metric.get_metric()
    logger.info('Metrics on validation dataset: %s' % str(metric_result))


def train_segmentor(data_loaders, id2label, model, optimizer, lr_scheduler, args):
    model.train()

    start_epoch = -1
    latest_checkpoint = os.path.join(args.save_dir, 'latest.pth')
    if args.auto_resume and os.path.isfile(latest_checkpoint):
        checkpoint = torch.load(latest_checkpoint)

        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        logger.info('Resume from epoch %d' % start_epoch)

    train_loader = data_loaders['train']
    for epoch in range(start_epoch + 1, args.epochs):
        for step, data_dict in enumerate(train_loader, 1):
            load_data_to_gpu(data_dict)
            out, loss = model(data_dict)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % args.log_iter_interval == 0:
                logger.info(
                    'Iter [%d/%d] in epoch [%d/%d] lr: %f, loss: %f' % (step, len(train_loader), epoch, args.epochs, lr_scheduler.get_last_lr()[0], loss.cpu().item()))

        lr_scheduler.step()

        # save checkpoint
        if args.auto_resume:
            save_checkpoint(epoch, model, optimizer, lr_scheduler, args.save_dir)

        # evaluation
        if epoch % args.eval_epoch_interval == 0:
            logger.info('Evaluate on epoch: %d' % epoch)
            evaluate(data_loaders['val'], model, id2label, args)


def main():
    # parse args
    args = parse_args()

    # load data
    train_dataset = WaymoDataset(args.data_dir, 'training')
    logger.info('Loaded %d train samples' % len(train_dataset))

    val_dataset = WaymoDataset(args.data_dir, 'validation')
    logger.info('Loaded %d validation samples' % len(val_dataset))

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True,
        num_workers=args.num_workers, collate_fn=train_dataset.collate_batch,
        drop_last=False, sampler=None, timeout=0
    )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=False,
        num_workers=args.num_workers, collate_fn=val_dataset.collate_batch,
        drop_last=False, sampler=None, timeout=0
    )

    data_loaders = {'train': train_loader, 'val': val_loader}

    # define model
    model = MVFNet(train_dataset)
    model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    # train and evaluation
    train_segmentor(data_loaders, train_dataset.id2label, model, optimizer, lr_scheduler, args)


if __name__ == '__main__':
    main()
