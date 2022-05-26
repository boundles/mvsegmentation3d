import mmcv
import glob
import os
import sys
import numpy as np
from tqdm import tqdm

from mmseg.apis import inference_segmentor, init_segmentor

if __name__ == '__main__':
    # args
    data_dir = sys.argv[1]
    work_dir = sys.argv[2]
    camera_id = int(sys.argv[3])
    split = sys.argv[4]

    # data dirs
    lidar_dir = os.path.join(data_dir, split, 'lidar')
    label_dir = os.path.join(data_dir, split, 'label')
    image_dir = os.path.join(data_dir, split, 'image', str(camera_id))
    feature_dir = os.path.join(data_dir, split, 'image_feature', str(camera_id))

    # init model
    config_file = os.path.join(work_dir, 'segformer_mit-b3_8x1_769x769_160k_waymo.py')
    checkpoint_file = os.path.join(work_dir, 'latest.pth')
    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

    # get test set frames
    target_dir = label_dir
    test_set_frames = None
    if split == 'testing':
        target_dir = lidar_dir
        test_set_frames = dict()
        with open(os.path.join(work_dir, '3d_semseg_test_set_frames.txt'), 'r') as fp:
            lines = fp.read().splitlines()
            for line in lines:
                pps = line.split(',')
                test_set_frames[pps[0] + '_' + pps[1]] = 1

    file_pathnames = glob.glob(os.path.join(target_dir, '*.npy'))
    feature_file_pathnames = []
    if test_set_frames:
        for file_pathname in file_pathnames:
            filename = os.path.basename(file_pathname)
            pps = filename.split('-')
            if (pps[0] + '_' + pps[1]) in test_set_frames:
                feature_file_pathnames.append(file_pathname)
        print('Total %d frames: , %d feature frames' % (len(test_set_frames), len(feature_file_pathnames)))
    else:
        feature_file_pathnames = file_pathnames

    for file_pathname in tqdm(feature_file_pathnames):
        # get file name from path
        file_name = os.path.basename(file_pathname).replace('.npy', '')
        # inference on image file
        image_feature = {}
        feature_file = os.path.join(feature_dir, file_name + '.npy')
        image_file = os.path.join(image_dir, file_name + '.png')
        try:
            img = mmcv.imread(image_file, channel_order='rgb')
        except:
            print('read image file: %s failed' % image_file)
            np.save(feature_file, image_feature)
            continue
        result = inference_segmentor(model, img)

        # extract feature from image feature map
        lidar_file = os.path.join(lidar_dir, file_name + '.npy')
        lidar = np.load(lidar_file)
        for i in range(lidar.shape[0]):
            point = lidar[i, :]
            if int(point[6]) == (camera_id + 1):
                image_feature[i] = result[0][:, int(point[8]), int(point[7])]
            elif int(point[9]) == (camera_id + 1):
                image_feature[i] = result[0][:, int(point[11]), int(point[10])]
        np.save(feature_file, image_feature)
