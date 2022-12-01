import torch
import torch.nn as nn
import torch.optim as optim

from mvseg3d.models import SPNet, Segformer, LovaszLoss, OHEMCrossEntropyLoss, WarmupPolyLR


def build_segmentor(cfg, dataset):
    if cfg.MODEL.SEGMENTOR == 'segformer':
        batching_info = []
        for single_batch_info in cfg.MODEL.BATCHING_INFO:
            new_single_batch_info = {}
            for lvl in single_batch_info:
                new_single_batch_info[int(lvl)] = single_batch_info[lvl]
            batching_info.append(new_single_batch_info)
        segmentor = Segformer(dataset=dataset, batching_info=batching_info, window_shape=cfg.MODEL.WINDOW_SHAPE,
                              depths=cfg.MODEL.DEPTHS, drop_path_rate=cfg.MODEL.DROP_PATH_RATE)
    elif cfg.MODEL.SEGMENTOR == 'spnet':
        segmentor = SPNet(dataset=dataset)
    else:
        raise NotImplementedError

    return segmentor


def build_criterion(cfg, dataset):
    losses = []
    for loss_name in cfg.MODEL.LOSSES:
        if loss_name == 'ce':
            criterion = nn.CrossEntropyLoss(ignore_index=dataset.ignore_index)
        elif loss_name == 'ohem_ce':
            criterion = OHEMCrossEntropyLoss(keep_ratio=cfg.MODEL.OHEM_KEEP_RATIO,
                                             ignore_index=dataset.ignore_index)
        elif loss_name == 'lovasz':
            criterion = LovaszLoss(ignore_index=dataset.ignore_index)
        else:
            raise NotImplementedError
        losses.append((criterion, cfg.MODEL.LOSSES[loss_name]))

    return losses


def build_optimizer(cfg, model):
    if cfg.TRAIN.OPTIMIZER == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    elif cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY,
            momentum=cfg.TRAIN.MOMENTUM
        )
    else:
        raise NotImplementedError

    return optimizer


def build_scheduler(cfg, optimizer, epochs, iters_per_epoch):
    if cfg.TRAIN.LR_SCHEDULER == 'cosine_annealing':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * iters_per_epoch)
    elif cfg.TRAIN.LR_SCHEDULER == 'warmup_poly_lr':
        lr_scheduler = WarmupPolyLR(optimizer, max_iters=epochs * iters_per_epoch, warmup_iters=iters_per_epoch)
    elif cfg.TRAIN.LR_SCHEDULER == 'cyclic':
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=cfg.TRAIN.CYCLIC_BASE_LR,
                                                         max_lr=cfg.TRAIN.CYCLIC_MAX_LR,
                                                         mode='exp_range', gamma=0.9999,
                                                         step_size_up=iters_per_epoch * 2.5)
    else:
        raise NotImplementedError

    return lr_scheduler
