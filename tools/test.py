import os
import time
import argparse
import numpy as np
import zlib
from tqdm import tqdm

import torch
import torch.optim

from mvseg3d.datasets.waymo_dataset import WaymoDataset
from mvseg3d.datasets import build_dataloader
from mvseg3d.models.segmentors.mvf import MVFNet
from mvseg3d.utils.logging import get_logger

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.protos import segmentation_metrics_pb2
from waymo_open_dataset.protos import segmentation_submission_pb2

TOP_LIDAR_ROW_NUM = 64
TOP_LIDAR_COL_NUM = 2650

def load_data_to_gpu(data_dict):
    for key, val in data_dict.items():
        if not isinstance(val, np.ndarray) or key == 'points_ri':
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

def get_range_index(points_ri):
    ri1_start_index = 0
    ri1_end_index = None
    ri2_start_index = None
    ri2_end_index = None
    for i in range(points_ri.shape[0]):
        if not ri1_end_index and i < (points_ri.shape[0] - 1) and points_ri[i, 0] != -1 and points_ri[
            i + 1, 0] == -1:
            ri1_end_index = i

        if not ri2_start_index and i > 0 and points_ri[i - 1, 0] == -1 and points_ri[i, 0] != -1:
            ri2_start_index = i

        if ri2_start_index and i < (points_ri.shape[0] - 1) and points_ri[i, 0] != -1 and points_ri[
            i + 1, 0] == -1:
            ri2_end_index = i

    return ri1_start_index, ri1_end_index, ri2_start_index, ri2_end_index

def compress_array(array: np.ndarray, is_int32: bool = False):
    """Compress a numpy array to ZLIP compressed serialized MatrixFloat/Int32.

    Args:
    array: A numpy array.
    is_int32: If true, use MatrixInt32, otherwise use MatrixFloat.

    Returns:
    The compressed bytes.
    """
    if is_int32:
        m = open_dataset.MatrixInt32()
    else:
        m = open_dataset.MatrixFloat()
        m.shape.dims.extend(list(array.shape))
        m.data.extend(array.reshape([-1]).tolist())
    return zlib.compress(m.SerializeToString())

def semseg_for_one_frame(model, data_dict):
    load_data_to_gpu(data_dict)
    with torch.no_grad():
        out = model(data_dict)
    pred_labels = torch.argmax(out, dim=1).cpu()

    # assign the dummy class to all valid points (in the range image)
    range_image_pred = np.zeros(
        (TOP_LIDAR_ROW_NUM, TOP_LIDAR_COL_NUM, 2), dtype=np.int32)
    range_image_pred_ri2 = np.zeros(
        (TOP_LIDAR_ROW_NUM, TOP_LIDAR_COL_NUM, 2), dtype=np.int32)

    points_ri = data_dict['points_ri']
    ri1_start_index, ri1_end_index, ri2_start_index, ri2_end_index = get_range_index(points_ri)
    for i in range(pred_labels.shape):
        if ri1_start_index <= i <= ri1_end_index:
            range_image_pred[points_ri[i, 1], points_ri[i, 0], 1] = pred_labels[i].item() + 1
        elif ri2_start_index <= i <= ri2_end_index:
            range_image_pred_ri2[points_ri[i, 1], points_ri[i, 0], 1] = pred_labels[i].item() + 1

    # construct the segmentationFrame proto.
    name_parts = data_dict['filename'][0].split('-')
    segmentation_frame = segmentation_metrics_pb2.SegmentationFrame()
    segmentation_frame.context_name = name_parts[0]
    segmentation_frame.frame_timestamp_micros = long(name_parts[1])
    laser_semseg = open_dataset.Laser()
    laser_semseg.name = open_dataset.LaserName.TOP
    laser_semseg.ri_return1.segmentation_label_compressed = compress_array(
        range_image_pred, is_int32=True)
    laser_semseg.ri_return2.segmentation_label_compressed = compress_array(
        range_image_pred_ri2, is_int32=True)
    segmentation_frame.segmentation_labels.append(laser_semseg)
    return segmentation_frame

def inference(data_loader, model, logger):
    logger('inference start!')
    model.eval()
    segmentation_frame_list = segmentation_metrics_pb2.SegmentationFrameList()
    for step, data_dict in enumerate(tqdm(data_loader, 1)):
        segmentation_frame = semseg_for_one_frame(model, data_dict)
        segmentation_frame_list.frames.append(segmentation_frame)

    # create the submission file, which can be uploaded to the eval server.
    submission = segmentation_submission_pb2.SemanticSegmentationSubmission()
    submission.account_name = '835667385@qq.com'
    submission.unique_method_name = 'WNet'
    submission.affiliation = 'WPCLab'
    submission.authors.append('Darren Wang')
    submission.description = "A fusion-based method by WPCLab."
    submission.method_link = 'NA'
    submission.sensor_type = 1
    submission.number_past_frames_exclude_current = 2
    submission.number_future_frames_exclude_current = 0
    submission.inference_results.CopyFrom(segmentation_frame_list)

    output_filename = './wod_semseg_test_set_pred_submission.bin'
    f = open(output_filename, 'wb')
    f.write(submission.SerializeToString())
    f.close()

    logger('inference finished!')

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
    checkpoint = torch.load(os.path.join(args.save_dir, 'latest.pth'), map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    # inference
    inference(test_loader, model, logger)


if __name__ == '__main__':
    main()
