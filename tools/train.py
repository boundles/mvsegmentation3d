import os
import argparse
import numpy as np

import torch
import torch.optim

from mvseg3d.datasets.waymo_dataset import WaymoDataset
from mvseg3d.datasets import build_dataloader
from mvseg3d.models.segmentors.mvf import MVFNet
from mvseg3d.core.metrics import IOUMetric
from mvseg3d.utils.logging import get_logger
from mvseg3d.utils import distributed_utils

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
    parser.add_argument('--data_dir', type=str, help='the data directory')
    parser.add_argument('--save_dir', type=str, help='the saved directory')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_gpus', default=1, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr', default=0.0125, type=float)
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--no_validate', action='store_true', help='whether not to evaluate the checkpoint during training')
    parser.add_argument('--eval_epoch_interval', default=2, type=int)
    parser.add_argument('--log_iter_interval', default=5, type=int)
    parser.add_argument('--auto_resume', action='store_true', help='resume from the latest checkpoint automatically')
    args = parser.parse_args()

    # calculate the batch size
    batch_size = args.num_gpus * args.batch_size
    logger.info(f'Training with {args.num_gpus} GPU(s) with {args.batch_size} '
                f'samples per GPU. The total batch size is {batch_size}.')

    # calculate the num of workers
    num_workers = args.num_gpus * args.num_workers

    if batch_size != args.batch_size:
        # scale LR with
        # [linear scaling rule](https://arxiv.org/abs/1706.02677)
        scaled_lr = (batch_size / args.batch_size) * args.lr
        logger.info('LR has been automatically scaled '
                    f'from {args.lr} to {scaled_lr}')
        args.lr = scaled_lr
        args.batch_size = batch_size
        args.num_workers = num_workers
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

def evaluate(args, data_loader, model, id2label, epoch):
    iou_metric = IOUMetric(id2label)
    model.eval()
    for step, data_dict in enumerate(data_loader, 1):
        load_data_to_gpu(data_dict)
        with torch.no_grad():
            out, loss = model(data_dict)
        if step % args.log_iter_interval == 0:
            logger.info(
                'Evaluate on epoch %d - Iter [%d/%d] loss: %f' % (epoch, step, len(data_loader), loss.cpu().item()))

        pred_labels = torch.argmax(out, dim=1).cpu()
        gt_labels = data_dict['labels'].cpu()
        iou_metric.add(pred_labels, gt_labels)

    metric_result = iou_metric.get_metric()
    logger.info('Metrics on validation dataset: %s' % str(metric_result))

def train_epoch(args, data_loader, model, optimizer, epoch, rank):
    for step, data_dict in enumerate(data_loader, 1):
        load_data_to_gpu(data_dict)
        out, loss = model(data_dict)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if rank == 0 and step % args.log_iter_interval == 0:
            try:
                cur_lr = float(optimizer.lr)
            except:
                cur_lr = optimizer.param_groups[0]['lr']
            logger.info(
                'Train - Epoch [%d/%d] Iter [%d/%d] lr: %f, loss: %f' % (epoch, args.epochs - 1, step, len(data_loader), cur_lr, loss.cpu().item()))

def train_segmentor(args, data_loaders, train_sampler, id2label, model, optimizer, lr_scheduler, rank):
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

    for epoch in range(start_epoch + 1, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train_epoch(args, data_loaders['train'], model, optimizer, epoch, rank)

        lr_scheduler.step()

        # evaluate on validation set
        if not args.no_validate and epoch % args.eval_epoch_interval == 0:
            evaluate(data_loaders['val'], model, id2label, args)

        # save checkpoint
        if rank == 0 and args.auto_resume:
            save_checkpoint(epoch, model, optimizer, lr_scheduler, args.save_dir)


def main():
    # parse args
    args = parse_args()

    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        distributed_utils.init_dist(args.launcher)
        # gpu_ids is used to calculate iter when resuming checkpoint
        rank, world_size = distributed_utils.get_dist_info()

    # load data
    train_dataset = WaymoDataset(args.data_dir, 'training', use_image_feature=True)
    logger.info('Loaded %d train samples' % len(train_dataset))

    val_dataset = WaymoDataset(args.data_dir, 'validation', use_image_feature=True)
    logger.info('Loaded %d validation samples' % len(val_dataset))

    train_set, train_loader, train_sampler = build_dataloader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        dist=distributed,
        num_workers=args.num_workers,
        training=True)

    val_set, val_loader, sampler = build_dataloader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        dist=distributed,
        num_workers=args.num_workers,
        training=False)

    data_loaders = {'train': train_loader, 'val': val_loader}

    # define model
    model = MVFNet(train_dataset)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank % torch.cuda.device_count()])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    # train and evaluation
    train_segmentor(args, data_loaders, train_sampler, train_dataset.id2label, model, lr_scheduler, rank)


if __name__ == '__main__':
    main()
