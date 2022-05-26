import torch.optim as optim

from .voxel_encoders import MeanVFE
from .backbones import SparseUnet


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


__all__ = ['MeanVFE', 'SparseUnet', build_optimizer]
