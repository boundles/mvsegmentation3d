import os
import time
import argparse

import torch
import torch.optim

from mvseg3d.datasets.waymo_dataset import WaymoDataset
from mvseg3d.datasets import build_dataloader
from mvseg3d.models.segmentors.spnet import SPNet
from mvseg3d.core.metrics import IOUMetric
from mvseg3d.utils.logging import get_logger
from mvseg3d.utils.data_utils import load_data_to_gpu


def parse_args():
    parser = argparse.ArgumentParser(description='Test a 3d segmentor')
    parser.add_argument('--data_dir', type=str, help='the data directory')
    parser.add_argument('--save_dir', type=str, help='the saved directory')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cudnn_benchmark', action='store_true', default=False, help='whether to use cudnn')
    parser.add_argument('--log_iter_interval', default=5, type=int)
    args = parser.parse_args()

    return args

def evaluate(args, data_loader, model, class_names, logger):
    iou_metric = IOUMetric(class_names)
    model.eval()
    for step, data_dict in enumerate(data_loader, 1):
        load_data_to_gpu(data_dict)
        with torch.no_grad():
            out, loss = model(data_dict)
        if step % args.log_iter_interval == 0:
            logger.info(
                'Evaluate - Iter [%d/%d] loss: %f' % ( step, len(data_loader), loss.cpu().item()))

        pred_labels = torch.argmax(out, dim=1).cpu()
        gt_labels = data_dict['labels'].cpu()
        iou_metric.add(pred_labels, gt_labels)

    metric_result = iou_metric.get_metric()
    logger.info('Metrics on validation dataset: %s' % str(metric_result))

def main():
    # parse args
    args = parse_args()

    # set cudnn_benchmark
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    # create logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.save_dir, f'{timestamp}.log')
    logger = get_logger("mvseg3d", log_file)

    # load data
    val_dataset = WaymoDataset(args.data_dir, 'validation', use_image_feature=True)
    logger.info('Loaded %d validation samples' % len(val_dataset))

    val_set, val_loader, sampler = build_dataloader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        dist=False,
        num_workers=args.num_workers,
        training=False)

    # define model
    model = SPNet(val_dataset).cuda()
    checkpoint = torch.load(os.path.join(args.save_dir, 'latest.pth'), map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    # evaluation
    evaluate(args, val_loader, model, val_dataset.class_names, logger)


if __name__ == '__main__':
    main()
