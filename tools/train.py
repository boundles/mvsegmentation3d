import os
import time
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
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--no_validate', action='store_true', help='whether not to evaluate the checkpoint during training')
    parser.add_argument('--eval_epoch_interval', default=2, type=int)
    parser.add_argument('--log_iter_interval', default=5, type=int)
    parser.add_argument('--auto_resume', action='store_true', help='resume from the latest checkpoint automatically')
    args = parser.parse_args()

    return args

def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu

def save_checkpoint(epoch, model, optimizer, lr_scheduler, save_dir, logger):
    logger.info('Save checkpoint at epoch %d' % epoch)
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_state = model_state_to_cpu(model.module.state_dict())
    else:
        model_state = model.state_dict()

    checkpoint = {
        "model": model_state,
        'optimizer': optimizer.state_dict(),
        "epoch": epoch,
        'lr_scheduler': lr_scheduler.state_dict()
    }

    torch.save(checkpoint, os.path.join(save_dir, 'epoch_%s.pth' % str(epoch)))
    torch.save(checkpoint, os.path.join(save_dir, 'latest.pth'))

def evaluate(args, data_loader, model, id2label, epoch, logger):
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

def train_epoch(args, data_loader, model, optimizer, epoch, rank, logger):
    model.train()
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

def train_segmentor(args, start_epoch, data_loaders, train_sampler, id2label, model, optimizer, lr_scheduler, rank, logger):
    for epoch in range(start_epoch + 1, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train_epoch(args, data_loaders['train'], model, optimizer, epoch, rank, logger)

        lr_scheduler.step()

        # save checkpoint
        if rank == 0 and args.auto_resume:
            save_checkpoint(epoch, model, optimizer, lr_scheduler, args.save_dir, logger)

        # evaluate on validation set
        if rank == 0 and not args.no_validate and epoch % args.eval_epoch_interval == 0:
            evaluate(args, data_loaders['val'], model, id2label, epoch, logger)

def main():
    # parse args
    args = parse_args()

    # whether to distributed training
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        distributed_utils.init_dist(args.launcher)
        # gpu_ids is used to calculate iter when resuming checkpoint
        rank, world_size = distributed_utils.get_dist_info()

    # create saved directory
    os.makedirs(args.save_dir, exist_ok=True)

    # create logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.save_dir, f'{timestamp}.log')
    logger = get_logger("mvseg3d", log_file)

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
        dataset=val_dataset,
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

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = -1
    latest_checkpoint = os.path.join(args.save_dir, 'latest.pth')
    if args.auto_resume and os.path.isfile(latest_checkpoint):
        loc_type = torch.device('cpu') if distributed else None
        checkpoint = torch.load(latest_checkpoint, map_location=loc_type)

        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        logger.info('Resume from epoch %d' % start_epoch)

    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank % torch.cuda.device_count()])

    # train and evaluation
    train_segmentor(args, start_epoch, data_loaders, train_sampler, train_dataset.id2label, model, optimizer, lr_scheduler, rank, logger)


if __name__ == '__main__':
    main()
