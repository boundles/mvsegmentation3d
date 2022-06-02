import torch
import torch.nn as nn
import torch.optim as optim

from .voxel_encoders import MeanVFE
from .backbones import SparseUnet
from .losses import LovaszLoss


def build_optimizer(cfg, model):
    if cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    elif cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY,
            momentum=cfg.TRAIN.MOMENTUM
        )
    else:
        raise NotImplementedError

    return optimizer


def build_criterion(cfg, dataset):
    if dataset.class_weight:
        weight = torch.FloatTensor(dataset.class_weight)
    else:
        weight = None

    if cfg.MODEL.LOSS == 'ce':
        criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=dataset.ignore_index)
    elif cfg.MODEL.LOSS == 'lovasz':
        criterion = LovaszLoss(ignore_index=dataset.ignore_index)
    else:
        raise NotImplementedError

    return criterion


__all__ = ['MeanVFE', 'SparseUnet', build_optimizer, build_criterion]
