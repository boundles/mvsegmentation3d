import os
import glob
import numpy as np
import cv2

from torch.utils.data import Dataset, DataLoader
import tensorflow.compat.v1 as tf

from waymo_open_dataset.utils import range_image_utils, transform_utils, frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset


class WaymoParser(Dataset):
    def __init__(self,
                 load_dir,
                 save_dir,
                 test_mode=False):
        # turn on eager execution
        tf.enable_eager_execution()

        self.prefix = ''
        self.filter_no_label_zone_points = True

        self.load_dir = load_dir
        self.save_dir = save_dir
        self.test_mode = test_mode

        self.tfrecord_pathnames = sorted(glob.glob(os.path.join(self.load_dir, '*.tfrecord')))

        self.label_save_dir = f'{self.save_dir}/label'
        self.image_save_dir = f'{self.save_dir}/image'
        self.calib_save_dir = f'{self.save_dir}/calib'
        self.point_cloud_save_dir = f'{self.save_dir}/lidar'
        self.pose_save_dir = f'{self.save_dir}/pose'

        self.create_folder()

    def __getitem__(self, index):
        """Convert action for single file.
        Args:
            file_idx (int): Index of the file to be converted.
        """
        pathname = self.tfrecord_pathnames[index]
        dataset = tf.data.TFRecordDataset(pathname, compression_type='')

        print('Processing ' + pathname)
        for frame_idx, data in enumerate(dataset):
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            self.save_image(frame, index, frame_idx)
            self.save_calib(frame, index, frame_idx)
            self.save_lidar_and_label(frame, index, frame_idx)
            self.save_pose(frame, index, frame_idx)

        return True

    def __len__(self):
        """Length of the filename list."""
        return len(self.tfrecord_pathnames)

    def save_image(self, frame, file_idx, frame_idx):
        """Parse and save the images in png format.
        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        for img in frame.images:
            img_path = f'{self.image_save_dir}/{str(img.name - 1)}/' + \
                       f'{self.prefix}{str(file_idx).zfill(3)}' + \
                       f'{str(frame_idx).zfill(3)}.png'
            img = tf.image.decode_jpeg(img.image)
            cv2.imwrite(img_path, img.numpy())

    def save_calib(self, frame, file_idx, frame_idx):
        """Parse and save the calibration data.
        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        # waymo front camera to kitti reference camera
        T_front_cam_to_ref = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0],
                                       [1.0, 0.0, 0.0]])
        camera_calibs = []
        R0_rect = [f'{i:e}' for i in np.eye(3).flatten()]
        Tr_velo_to_cams = []
        calib_context = ''

        for camera in frame.context.camera_calibrations:
            # extrinsic parameters
            T_cam_to_vehicle = np.array(camera.extrinsic.transform).reshape(
                4, 4)
            T_vehicle_to_cam = np.linalg.inv(T_cam_to_vehicle)
            Tr_velo_to_cam = \
                self.cart_to_homo(T_front_cam_to_ref) @ T_vehicle_to_cam
            if camera.name == 1:  # FRONT = 1, see dataset.proto for details
                self.T_velo_to_front_cam = Tr_velo_to_cam.copy()
            Tr_velo_to_cam = Tr_velo_to_cam[:3, :].reshape((12,))
            Tr_velo_to_cams.append([f'{i:e}' for i in Tr_velo_to_cam])

            # intrinsic parameters
            camera_calib = np.zeros((3, 4))
            camera_calib[0, 0] = camera.intrinsic[0]
            camera_calib[1, 1] = camera.intrinsic[1]
            camera_calib[0, 2] = camera.intrinsic[2]
            camera_calib[1, 2] = camera.intrinsic[3]
            camera_calib[2, 2] = 1
            camera_calib = list(camera_calib.reshape(12))
            camera_calib = [f'{i:e}' for i in camera_calib]
            camera_calibs.append(camera_calib)

        # all camera ids are saved as id-1 in the result because
        # camera 0 is unknown in the proto
        for i in range(5):
            calib_context += 'P' + str(i) + ': ' + \
                             ' '.join(camera_calibs[i]) + '\n'
        calib_context += 'R0_rect' + ': ' + ' '.join(R0_rect) + '\n'
        for i in range(5):
            calib_context += 'Tr_velo_to_cam_' + str(i) + ': ' + \
                             ' '.join(Tr_velo_to_cams[i]) + '\n'

        with open(
                f'{self.calib_save_dir}/{self.prefix}' +
                f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}.txt',
                'w+') as fp_calib:
            fp_calib.write(calib_context)
            fp_calib.close()

    @staticmethod
    def convert_range_image_to_point_cloud_labels(frame,
                                                  range_images,
                                                  segmentation_labels,
                                                  ri_index=0):
        """Convert segmentation labels from range images to point clouds.

        Args:
          frame: open dataset frame
          range_images: A dict of {laser_name, [range_image_first_return,
             range_image_second_return]}.
          segmentation_labels: A dict of {laser_name, [range_image_first_return,
             range_image_second_return]}.
          ri_index: 0 for the first return, 1 for the second return.

        Returns:
          point_labels: {[N, 2]} list of 3d lidar points's segmentation labels. 0 for
            points that are not labeled.
        """
        calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
        point_labels = []
        for c in calibrations:
            range_image = range_images[c.name][ri_index]
            range_image_tensor = tf.reshape(
                tf.convert_to_tensor(range_image.data), range_image.shape.dims)
            range_image_mask = range_image_tensor[..., 0] > 0

            if c.name in segmentation_labels:
                sl = segmentation_labels[c.name][ri_index]
                sl_tensor = tf.reshape(tf.convert_to_tensor(sl.data), sl.shape.dims)
                sl_points_tensor = tf.gather_nd(sl_tensor, tf.where(range_image_mask))
            else:
                num_valid_point = tf.math.reduce_sum(tf.cast(range_image_mask, tf.int32))
                sl_points_tensor = tf.zeros([num_valid_point, 2], dtype=tf.int32)

            point_labels.append(sl_points_tensor.numpy())
        return point_labels

    def save_lidar_and_label(self, frame, file_idx, frame_idx):
        """Parse and save the lidar data in psd format.
        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        range_images, camera_projections, segmentation_labels, range_image_top_pose = \
            frame_utils.parse_range_image_and_camera_projection(frame)

        # points of first return
        points_0, cp_points_0 = frame_utils.convert_range_image_to_point_cloud(
            frame, range_images, camera_projections, range_image_top_pose, ri_index=0, keep_polar_features=True)

        # points of second return
        points_1, cp_points_2 = frame_utils.convert_range_image_to_point_cloud(
            frame, range_images, camera_projections, range_image_top_pose, ri_index=1, keep_polar_features=True)

        # point cloud with 6-dim: x, y, z, intensity, and elongation, range
        point_cloud = np.concatenate([points_0, points_1], axis=0)
        point_cloud = point_cloud[:, [0, 1, 2, 4, 5, 3]]

        pc_path = f'{self.point_cloud_save_dir}/{self.prefix}' + \
                  f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}'
        np.save(pc_path, point_cloud)

        if len(segmentation_labels) > 0:
            # point labels of first return
            point_labels_0 = self.convert_range_image_to_point_cloud_labels(
                frame, range_images, segmentation_labels, ri_index=0)

            # point labels of second return
            point_labels_1 = self.convert_range_image_to_point_cloud_labels(
                frame, range_images, segmentation_labels, ri_index=1)

            point_labels = np.concatenate([point_labels_0, point_labels_1], axis=0)
            point_labels = point_labels[:, 1]

            # convert 0 to 255 for ignore label
            point_labels -= 1
            point_labels[point_labels == -1] = 255

            label_path = f'{self.label_save_dir}/{self.prefix}' + \
                         f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}'
            np.save(label_path, point_labels)

    def save_pose(self, frame, file_idx, frame_idx):
        """Parse and save the pose data.
        Note that SDC's own pose is not included in the regular training
        of KITTI dataset. KITTI raw dataset contains ego motion files
        but are not often used. Pose is important for algorithms that
        take advantage of the temporal information.
        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        pose = np.array(frame.pose.transform).reshape(4, 4)
        np.savetxt(
            os.path.join(f'{self.pose_save_dir}/{self.prefix}' +
                         f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}.txt'),
            pose)

    def create_folder(self):
        """Create folder for data preprocessing."""
        if not self.test_mode:
            dir_list = [
                self.point_cloud_save_dir, self.label_save_dir,
                self.calib_save_dir, self.pose_save_dir,
                self.image_save_dir
            ]
        else:
            dir_list = [
                self.calib_save_dir, self.point_cloud_save_dir,
                self.pose_save_dir, self.image_save_dir
            ]
        for d in dir_list:
            if not os.path.exists(d):
                os.makedirs(d)
        for i in range(5):
            if not os.path.exists(f'{self.image_save_dir}/{str(i)}'):
                os.makedirs(f'{self.image_save_dir}/{str(i)}')

    def cart_to_homo(self, mat):
        """Convert transformation matrix in Cartesian coordinates to
        homogeneous format.
        Args:
            mat (np.ndarray): Transformation matrix in Cartesian.
                The input matrix shape is 3x3 or 3x4.
        Returns:
            np.ndarray: Transformation matrix in homogeneous format.
                The matrix shape is 4x4.
        """
        ret = np.eye(4)
        if mat.shape == (3, 3):
            ret[:3, :3] = mat
        elif mat.shape == (3, 4):
            ret[:3, :] = mat
        else:
            raise ValueError(mat.shape)
        return ret


if __name__ == '__main__':
    split = 'validation'
    raw_data_dir = os.path.join('/nfs/s3_common_dataset/waymo_perception_v1.3', split)
    parsed_data_dir = os.path.join('/nfs/volume-807-2/waymo_open_dataset_v_1_3_0', split)
    parser = WaymoParser(raw_data_dir, parsed_data_dir)
    print('Parse Started!')
    data_loader = DataLoader(parser, num_workers=8)
    for step, data in enumerate(data_loader):
        pass
    print('Parse Finished!')
