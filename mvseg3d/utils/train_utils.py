import math
import random
from typing import List
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import torch.distributed as dist

from mvseg3d.models import LovaszLoss, OHEMCrossEntropyLoss, DiceLoss
from mvseg3d.utils.distributed_utils import get_dist_info


def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.
    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.
    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def _get_warmup_factor_at_iter(
    method: str, iter: int, warmup_iters: int, warmup_factor: float
) -> float:
    """
    Return the learning rate warmup factor at a specific iteration.
    See :paper:`ImageNet in 1h` for more details.
    Args:
        method (str): warmup method; either "constant" or "linear".
        iter (int): iteration at which to calculate the warmup factor.
        warmup_iters (int): the number of warmup iterations.
        warmup_factor (float): the base warmup factor (the meaning changes according
            to the method used).
    Returns:
        float: the effective warmup factor at the given iteration.
    """
    if iter >= warmup_iters:
        return 1.0

    if method == "constant":
        return warmup_factor
    elif method == "linear":
        alpha = iter / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    else:
        raise ValueError("Unknown warmup method: {}".format(method))

# NOTE: PyTorch's LR scheduler interface uses names that assume the LR changes
# only on epoch boundaries. We typically use iteration based schedules instead.
# As a result, "epoch" (e.g., as in self.last_epoch) should be understood to mean
# "iteration" instead.
class WarmupPolyLR(_LRScheduler):
    """
    Poly learning rate schedule used to train DeepLab.
    Paper: DeepLab: Semantic Image Segmentation with Deep Convolutional Nets,
        Atrous Convolution, and Fully Connected CRFs.
    Reference: https://github.com/tensorflow/models/blob/21b73d22f3ed05b650e85ac50849408dd36de32e/research/deeplab/utils/train_utils.py#L337  # noqa
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_iters: int,
        warmup_factor: float = 0.001,
        warmup_iters: int = 1000,
        warmup_method: str = "linear",
        last_epoch: int = -1,
        power: float = 0.9,
        constant_ending: float = 0.0,
    ):
        self.max_iters = max_iters
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.power = power
        self.constant_ending = constant_ending
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        if self.constant_ending > 0 and warmup_factor == 1.0:
            # Constant ending lr.
            if (
                math.pow((1.0 - self.last_epoch / self.max_iters), self.power)
                < self.constant_ending
            ):
                return [base_lr * self.constant_ending for base_lr in self.base_lrs]
        return [
            base_lr * warmup_factor * math.pow((1.0 - self.last_epoch / self.max_iters), self.power)
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()

def build_criterion(cfg, dataset):
    if dataset.class_weight:
        weight = torch.FloatTensor(dataset.class_weight).cuda()
    else:
        weight = None

    losses = []
    for loss_name in cfg.MODEL.LOSSES:
        if loss_name == 'ce':
            criterion = OHEMCrossEntropyLoss(class_weight=weight, keep_ratio=cfg.MODEL.OHEM_KEEP_RATIO,
                                             ignore_index=dataset.ignore_index)
        elif loss_name == 'dice':
            criterion = DiceLoss(class_weight=weight, ignore_index=dataset.ignore_index)
        elif loss_name == 'lovasz':
            criterion = LovaszLoss(class_weight=weight, ignore_index=dataset.ignore_index)
        else:
            raise NotImplementedError
        losses.append((criterion, cfg.MODEL.LOSSES[loss_name]))

    return losses

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