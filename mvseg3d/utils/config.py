import numpy as np
from easydict import EasyDict as edict


__C = edict()
cfg = __C

# dataset config
__C.DATASET = edict()
__C.DATASET.USE_MULTI_SWEEPS = False
__C.DATASET.MAX_NUM_SWEEPS = 5
__C.DATASET.NUM_SWEEPS = 3
__C.DATASET.USE_CYLINDER = False
__C.DATASET.POINT_CLOUD_RANGE = [-75.2, -75.2, -2, 75.2, 75.2, 5.2]
__C.DATASET.VOXEL_SIZE = [0.1, 0.1, 0.1]
__C.DATASET.MAX_NUM_POINTS = 10
__C.DATASET.MAX_VOXELS = {'train': 90000, 'test': 150000}
__C.DATASET.DIM_POINT = 6
__C.DATASET.USE_IMAGE_FEATURE = True
__C.DATASET.DIM_IMAGE_FEATURE = 28
__C.DATASET.NUM_CLASSES = 22
__C.DATASET.CLASS_NAMES = []
__C.DATASET.CLASS_WEIGHT = []
__C.DATASET.IGNORE_INDEX = 255

__C.DATASET.AUG_DATA = True
__C.DATASET.AUG_ROT_RANGE = [-0.78539816, 0.78539816]
__C.DATASET.AUG_SCALE_RANGE = [0.95, 1.05]
__C.DATASET.AUG_SAMPLE_RATIO = 0.95
__C.DATASET.AUG_SAMPLE_RANGE = 50.0
__C.DATASET.AUG_DROP_RATIO = 0.5

# model config
__C.MODEL = edict()
__C.MODEL.LOSSES = {'ohem_ce': 1.0, 'lovasz': 1.0}
__C.MODEL.OHEM_KEEP_RATIO = 0.3
__C.MODEL.AUX_LOSS_WEIGHT = 0.4

# training config
__C.TRAIN = edict()
__C.TRAIN.OPTIMIZER = 'adam'
__C.TRAIN.LR = 0.001
__C.TRAIN.WEIGHT_DECAY = 0.01
__C.TRAIN.MOMENTUM = 0.9
__C.TRAIN.LR_SCHEDULER = 'warmup_poly_lr'
__C.TRAIN.CYCLIC_BASE_LR = 0.001
__C.TRAIN.CYCLIC_MAX_LR = 0.1


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

    _merge_a_into_b(yaml_cfg, __C)


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))
        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]), type(v), k))
        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(type(value), type(d[subkey]))
        d[subkey] = value


def save_config_to_file(cfg, pre='cfg', logger=None):
    for key, val in cfg.items():
        if isinstance(cfg[key], edict):
            if logger is not None:
                logger.info('\n%s.%s = edict()' % (pre, key))
            else:
                print('\n%s.%s = edict()' % (pre, key))
            save_config_to_file(cfg[key], pre=pre + '.' + key, logger=logger)
            continue

        if logger is not None:
            logger.info('%s.%s: %s' % (pre, key, val))
        else:
            print('%s.%s: %s' % (pre, key, val))