import mmcv
import glob
import os
import sys
import numpy as np
from tqdm import tqdm

from mmseg.apis import inference_segmentor, init_segmentor


def extract_image_features(model, image_dir, filename):
    image_feature_maps = dict()
    for camera_id in range(5):
        image_file = os.path.join(image_dir, str(camera_id), filename + '.png')
        try:
            img = mmcv.imread(image_file, channel_order='rgb')
        except:
            print('read image file: %s failed' % image_file)
            continue

        result = inference_segmentor(model, img)
        image_feature_maps[camera_id] = result[0]
    return image_feature_maps


def get_image_list(work_dir, split, pathnames_file):
    with open(pathnames_file, 'r') as fp:
        pathnames = fp.read().splitlines()

    test_set_frames = None
    if split == 'testing':
        test_set_frames = dict()
        with open(os.path.join(work_dir, '3d_semseg_test_set_frames.txt'), 'r') as fp:
            lines = fp.read().splitlines()
            for line in lines:
                infos = line.split(',')
                test_set_frames[(infos[0], infos[1])] = 1

    image_pathnames = []
    if test_set_frames:
        for pathname in pathnames:
            filename = os.path.basename(pathname)
            infos = filename.split('-')
            if (infos[0], infos[1]) in test_set_frames:
                image_pathnames.append(pathname)
        print('Total %d frames: , %d image frames' % (len(test_set_frames), len(image_pathnames)))
    else:
        image_pathnames = pathnames
    return image_pathnames


if __name__ == '__main__':
    # args
    data_dir = sys.argv[1]
    work_dir = sys.argv[2]
    split = sys.argv[3]
    pathnames_file = sys.argv[4]

    # data dirs
    lidar_dir = os.path.join(data_dir, split, 'lidar')
    image_dir = os.path.join(data_dir, split, 'image')
    feature_dir = os.path.join(data_dir, split, 'image_feature')

    # init model
    config_file = os.path.join(work_dir, 'segformer_mit-b3_8x1_769x769_160k_waymo.py')
    checkpoint_file = os.path.join(work_dir, 'latest.pth')
    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

    # extract feature from image paths
    image_pathnames = get_image_list(work_dir, split, pathnames_file)
    for image_pathname in tqdm(image_pathnames):
        # get file name from path
        filename = os.path.basename(image_pathname).replace('.npy', '')
        # inference on image file
        image_feature_maps = extract_image_features(model, image_dir, filename)

        # query point features from image feature map
        point_image_features = dict()
        feature_file = os.path.join(feature_dir, filename + '.npy')
        lidar_file = os.path.join(lidar_dir, filename + '.npy')
        lidar = np.load(lidar_file)
        for i in range(lidar.shape[0]):
            point = lidar[i, :]
            camera_id0 = int(point[6]) - 1
            image_x0 = int(point[7])
            image_y0 = int(point[8])
            camera_id1 = int(point[9]) - 1
            image_x1 = int(point[10])
            image_y1 = int(point[11])
            if camera_id0 in image_feature_maps:
                point_image_features[i] = image_feature_maps[camera_id0][:, image_y0, image_x0]
            elif camera_id1 in image_feature_maps:
                point_image_features[i] = image_feature_maps[camera_id1][:, image_y1, image_x1]
        np.save(feature_file, point_image_features)
