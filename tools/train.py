import os
import time
import argparse

import numpy as np

import torch
import torch.optim
import torch.distributed as dist

from spconv.core import ConvAlgo
from spconv.pytorch import ops

from mvseg3d.datasets.waymo_dataset import WaymoDataset
from mvseg3d.datasets import build_dataloader
from mvseg3d.models.segmentors.spnet import SPNet
from mvseg3d.core.metrics import IOUMetric
from mvseg3d.utils.logging import get_root_logger
from mvseg3d.utils import distributed_utils
from mvseg3d.utils.config import cfg, cfg_from_file
from mvseg3d.utils.data_utils import load_data_to_gpu
from mvseg3d.utils.train_utils import build_criterion, build_optimizer, build_scheduler, set_random_seed, \
    init_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Train a 3d segmentor')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--data_dir', type=str, help='the data directory')
    parser.add_argument('--save_dir', type=str, help='the saved directory')
    parser.add_argument('--pretrained_path', type=str, default=None, help='pretrained_path')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--cudnn_benchmark', action='store_true', default=False, help='whether to use cudnn')
    parser.add_argument('--deterministic', action='store_true', default=False, help='whether to use deterministic')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--no_validate', action='store_true', help='whether not to evaluate the checkpoint during training')
    parser.add_argument('--eval_epoch_interval', default=2, type=int)
    parser.add_argument('--log_iter_interval', default=10, type=int)
    parser.add_argument('--auto_resume', action='store_true', help='resume from the latest checkpoint automatically')
    args = parser.parse_args()

    return args

def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu

def save_checkpoint(model, optimizer, lr_scheduler, save_dir, epoch, logger):
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

def compute_loss(pred_result, data_dict, criterion):
    loss = 0

    point_gt_labels = data_dict['labels']
    point_pred_labels = pred_result['out']
    for loss_func, loss_weight in criterion:
        loss += loss_func(point_pred_labels, point_gt_labels) * loss_weight

    if 'aux_out' in pred_result:
        voxel_pred_labels = pred_result['aux_out']
        voxel_gt_labels = data_dict['voxel_labels']
        for loss_func, loss_weight in criterion:
            loss += cfg.MODEL.AUX_LOSS_WEIGHT * loss_func(voxel_pred_labels, voxel_gt_labels) * loss_weight

    return loss

def prepare_aux_targets(batch_dict):
    batch_size = batch_dict['batch_size']
    spatial_shapes = [[1504, 1504, 72], [752, 752, 36], [376, 376, 18], [188, 188, 9]]
    voxel_labels = [batch_dict['voxel_labels']]
    voxel_indices = [batch_dict['voxel_coords']]
    for i in range(len(spatial_shapes) - 1):
        out_inds, indice_pairs, _ = ops.get_indice_pairs(
            voxel_indices[-1].int(),
            batch_size=batch_size,
            spatial_shape=spatial_shapes[i],
            algo=ConvAlgo.Native,
            ksize=[3, 3, 3],
            stride=[2, 2, 2],
            padding=[1, 1, 1],
            dilation=[1, 1, 1],
            out_padding=[0, 0, 0]
        )

        voxel_labels_np = voxel_labels[-1].cpu().numpy()
        indice_pairs_np = indice_pairs.cpu().numpy()

        label_size = 256
        nfilters = 27
        voxel_label_counter = dict()
        for j in range(indice_pairs_np.shape[-1]):
            for filter_offset in range(nfilters):
                i_ind = indice_pairs_np[:, :, j][0][filter_offset]
                o_ind = indice_pairs_np[:, :, j][1][filter_offset]
                if i_ind != -1 and o_ind != -1:
                    if o_ind not in voxel_label_counter:
                        counter = np.zeros((label_size,), dtype=np.uint16)
                        counter[voxel_labels_np[i_ind]] += 1
                        voxel_label_counter[o_ind] = counter
                    else:
                        counter = voxel_label_counter[o_ind]
                        counter[voxel_labels_np[i_ind]] += 1
                        voxel_label_counter[o_ind] = counter

        voxel_labels_downsample = np.ones(out_inds.shape[0], dtype=np.uint8) * 255
        for voxel_id in voxel_label_counter:
            counter = voxel_label_counter[voxel_id]
            voxel_labels_downsample[voxel_id] = np.argmax(counter)

        voxel_indices.append(out_inds.float())
        voxel_labels.append(torch.from_numpy(voxel_labels_downsample).to(voxel_labels[-1].device))

    batch_dict['aux_voxel_indices'] = voxel_indices
    batch_dict['aux_voxel_labels'] = voxel_labels

def evaluate(args, data_loader, model, criterion, class_names, epoch, logger):
    iou_metric = IOUMetric(class_names)
    model.eval()
    for step, data_dict in enumerate(data_loader, 1):
        load_data_to_gpu(data_dict)
        with torch.no_grad():
            result = model(data_dict)

        loss = compute_loss(result, data_dict, criterion)

        if step % args.log_iter_interval == 0:
            logger.info(
                'Evaluate on epoch %d - Iter [%d/%d] loss: %f' % (epoch, step, len(data_loader), loss.cpu().item()))

        pred_labels = torch.argmax(result['out'], dim=1).cpu()
        gt_labels = data_dict['labels'].cpu()
        iou_metric.add(pred_labels, gt_labels)

    metric_result = iou_metric.get_metric()
    logger.info('Metrics on validation dataset: %s' % str(metric_result))

def train_epoch(args, data_loader, model, criterion, optimizer, lr_scheduler, epoch, logger):
    model.train()
    for step, data_dict in enumerate(data_loader, 1):
        load_data_to_gpu(data_dict)

        prepare_aux_targets(data_dict)

        result = model(data_dict)

        loss = compute_loss(result, data_dict, criterion)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_scheduler.step()

        if step % args.log_iter_interval == 0:
            try:
                cur_lr = float(optimizer.lr)
            except:
                cur_lr = optimizer.param_groups[0]['lr']
            logger.info(
                'Train - Epoch [%d/%d] Iter [%d/%d] lr: %f, loss: %f' % (epoch, args.epochs, step, len(data_loader),
                                                                         cur_lr, loss.cpu().item()))

def train_segmentor(args, start_epoch, data_loaders, train_sampler, class_names, model,
                    criterion, optimizer, lr_scheduler, rank, logger):
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # current epoch
        cur_epoch = epoch + 1

        # train for one epoch
        train_epoch(args, data_loaders['train'], model, criterion, optimizer, lr_scheduler, cur_epoch, logger)

        # save checkpoint
        if rank == 0 and args.auto_resume:
            save_checkpoint(model, optimizer, lr_scheduler, args.save_dir, cur_epoch, logger)

        # evaluate on validation set
        if not args.no_validate and cur_epoch % args.eval_epoch_interval == 0:
            evaluate(args, data_loaders['val'], model, criterion, class_names, cur_epoch, logger)

def main():
    # parse args
    args = parse_args()

    # whether to distributed training
    if args.launcher == 'none':
        distributed = False
        rank = 0
    else:
        distributed = True
        distributed_utils.init_dist(args.launcher)
        # gpu_ids is used to calculate iter when resuming checkpoint
        rank, world_size = distributed_utils.get_dist_info()

    # set cudnn_benchmark
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    # create saved directory
    os.makedirs(args.save_dir, exist_ok=True)

    # create logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.save_dir, f'{timestamp}.log')
    logger = get_root_logger(name="mvseg3d", log_file=log_file)

    # set random seed
    seed = init_random_seed(args.seed)
    seed = seed + dist.get_rank()
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)

    # config
    cfg_from_file(args.cfg_file)
    logger.info(cfg)

    # load data
    train_dataset = WaymoDataset(cfg, args.data_dir, 'training')
    logger.info('Loaded %d train samples' % len(train_dataset))

    val_dataset = WaymoDataset(cfg, args.data_dir, 'validation')
    logger.info('Loaded %d validation samples' % len(val_dataset))

    train_set, train_loader, train_sampler = build_dataloader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        dist=distributed,
        num_workers=args.num_workers,
        seed=seed,
        training=True)

    val_set, val_loader, sampler = build_dataloader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        dist=distributed,
        num_workers=args.num_workers,
        seed=seed,
        training=False)

    data_loaders = {'train': train_loader, 'val': val_loader}

    # define model
    model = SPNet(train_dataset)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    # load pretrained model
    if args.pretrained_path:
        logger.info('Load pretrained model from %s' % args.pretrained_path)
        loc_type = torch.device('cpu') if distributed else None
        pretrained = torch.load(args.pretrained_path, map_location=loc_type)
        model.load_state_dict(pretrained['model'], strict=False)

    # optimizer and learning rate scheduler
    optimizer = build_optimizer(cfg, model)
    lr_scheduler = build_scheduler(cfg, optimizer, args.epochs, len(train_loader))

    # resume from checkpoint
    start_epoch = 0
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
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[rank % torch.cuda.device_count()],
                                                          find_unused_parameters=True)

    # loss function
    criterion = build_criterion(cfg, train_dataset)

    # train and evaluation
    train_segmentor(args, start_epoch, data_loaders, train_sampler, train_dataset.class_names,
                    model, criterion, optimizer, lr_scheduler, rank, logger)


if __name__ == '__main__':
    main()
