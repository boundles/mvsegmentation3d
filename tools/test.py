import os
import time
import argparse
import numpy as np

import torch
import torch.optim

from mvseg3d.datasets.waymo_dataset import WaymoDataset
from mvseg3d.datasets import build_dataloader
from mvseg3d.models.segmentors.mvf import MVFNet
from mvseg3d.utils.logging import get_logger


def load_data_to_gpu(data_dict):
    for key, val in data_dict.items():
        if not isinstance(val, np.ndarray) or key == 'points_indexing':
            continue
        else:
            if key in ['point_voxel_ids', 'labels']:
                data_dict[key] = torch.from_numpy(val).long().cuda()
            else:
                data_dict[key] = torch.from_numpy(val).float().cuda()
    return data_dict


def parse_args():
    parser = argparse.ArgumentParser(description='Test a 3d segmentor')
    parser.add_argument('--data_dir', type=str, help='the data directory')
    parser.add_argument('--save_dir', type=str, help='the saved directory')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--log_iter_interval', default=5, type=int)
    args = parser.parse_args()

    return args

def inference(data_loader, model, logger):
    model.eval()
    for step, data_dict in enumerate(data_loader, 1):
        load_data_to_gpu(data_dict)
        with torch.no_grad():
            out, loss = model(data_dict)

        pred_labels = torch.argmax(out, dim=1).cpu()

def main():
    # parse args
    args = parse_args()

    # create logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.save_dir, f'{timestamp}.log')
    logger = get_logger("mvseg3d", log_file)

    # load data
    test_dataset = WaymoDataset(args.data_dir, 'testing', use_image_feature=True, test_mode=True)
    logger.info('Loaded %d testing samples' % len(test_dataset))

    test_set, test_loader, sampler = build_dataloader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        dist=False,
        num_workers=args.num_workers,
        training=False)

    # define model
    model = MVFNet(test_dataset).cuda()
    checkpoint = torch.load(os.path.join(args.save_dir, 'latest.pth'), map_loccation='cpu')
    model.load_state_dict(checkpoint['model'])

    # inference
    inference(test_loader, model, logger)


if __name__ == '__main__':
    main()
