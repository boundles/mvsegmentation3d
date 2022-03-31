import os
import sys
import glob
import numpy as np
import cv2

import tensorflow.compat.v1 as tf
tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils, transform_utils, frame_utils
from waymo_open_dataset.utils.frame_utils import parse_range_image_and_camera_projection
from waymo_open_dataset import dataset_pb2 as open_dataset

from mvseg3d.utils.progressbar import track_parallel_progress


class WaymoConverter(object):
    def __init__(self,
                 load_dir,
                 save_dir,
                 workers=16,
                 test_mode=False):
        # turn on eager execution for older tensorflow versions
        if int(tf.__version__.split('.')[0]) < 2:
            tf.enable_eager_execution()

        self.prefix = ''
        self.selected_waymo_locations = None
        self.filter_no_label_zone_points = True

        self.load_dir = load_dir
        self.save_dir = save_dir
        self.workers = int(workers)
        self.test_mode = test_mode

        self.tfrecord_pathnames = sorted(glob.glob(os.path.join(self.load_dir, '*.tfrecord')))

        self.label_save_dir = f'{self.save_dir}/label'
        self.image_save_dir = f'{self.save_dir}/image'
        self.calib_save_dir = f'{self.save_dir}/calib'
        self.point_cloud_save_dir = f'{self.save_dir}/lidar'
        self.pose_save_dir = f'{self.save_dir}/pose'

        self.create_folder()

    def convert(self):
        """Convert action."""
        print('Start converting ...')
        track_parallel_progress(self.convert_one, range(len(self)),
                                self.workers)
        print('\nFinished ...')

    def convert_one(self, file_idx):
        """Convert action for single file.
        Args:
            file_idx (int): Index of the file to be converted.
        """
        pathname = self.tfrecord_pathnames[file_idx]
        dataset = tf.data.TFRecordDataset(pathname, compression_type='')

        for frame_idx, data in enumerate(dataset):
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            if (self.selected_waymo_locations is not None
                    and frame.context.stats.location
                    not in self.selected_waymo_locations):
                continue

            self.save_image(frame, file_idx, frame_idx)
            self.save_calib(frame, file_idx, frame_idx)
            self.save_lidar_and_label(frame, file_idx, frame_idx)
            self.save_pose(frame, file_idx, frame_idx)

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

    def save_lidar_and_label(self, frame, file_idx, frame_idx):
        """Parse and save the lidar data in psd format.
        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        range_images, camera_projections, segmentation_labels, range_image_top_pose = \
            parse_range_image_and_camera_projection(frame)

        # First return
        points_0, cp_points_0, point_labels_0, intensity_0, elongation_0 = \
            self.convert_range_image_to_point_cloud_labels(
                frame,
                range_images,
                segmentation_labels,
                camera_projections,
                range_image_top_pose,
                ri_index=0
            )
        points_0 = np.concatenate(points_0, axis=0)
        point_labels_0 = np.concatenate(point_labels_0, axis=0)
        intensity_0 = np.concatenate(intensity_0, axis=0)
        elongation_0 = np.concatenate(elongation_0, axis=0)

        # Second return
        points_1, cp_points_1, point_labels_1, intensity_1, elongation_1 = \
            self.convert_range_image_to_point_cloud_labels(
                frame,
                range_images,
                segmentation_labels,
                camera_projections,
                range_image_top_pose,
                ri_index=1
            )
        points_1 = np.concatenate(points_1, axis=0)
        point_labels_1 = np.concatenate(point_labels_1, axis=0)
        intensity_1 = np.concatenate(intensity_1, axis=0)
        elongation_1 = np.concatenate(elongation_1, axis=0)

        points = np.concatenate([points_0, points_1], axis=0)
        point_labels = np.concatenate([point_labels_0, point_labels_1], axis=0)
        intensity = np.concatenate([intensity_0, intensity_1], axis=0)
        elongation = np.concatenate([elongation_0, elongation_1], axis=0)
        timestamp = frame.timestamp_micros * np.ones_like(intensity)

        # concatenate x,y,z, intensity, elongation, timestamp (6-dim)
        point_cloud = np.column_stack(
            (points, intensity, elongation, timestamp))

        pc_path = f'{self.point_cloud_save_dir}/{self.prefix}' + \
                  f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}'
        np.save(pc_path, point_cloud)

        if frame.lasers[0].ri_return1.segmentation_label_compressed:
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

    def convert_range_image_to_point_cloud_labels(self,
                                                  frame,
                                                  range_images,
                                                  segmentation_labels,
                                                  camera_projections,
                                                  range_image_top_pose,
                                                  ri_index=0):
        """Convert range images to point cloud and semantic labels.
        Args:
            frame (:obj:`Frame`): Open dataset frame.
            range_images (dict): Mapping from laser_name to list of two
                range images corresponding with two returns.
            camera_projections (dict): Mapping from laser_name to list of two
                camera projections corresponding with two returns.
            range_image_top_pose (:obj:`Transform`): Range image pixel pose for
                top lidar.
            ri_index (int, optional): 0 for the first return,
                1 for the second return. Default: 0.
        Returns:
            tuple[list[np.ndarray]]: (List of points with shape [N, 3],
                camera projections of points with shape [N, 6], lables with shape [N, 1],
                intensity with shape [N, 1], elongation with shape [N, 1]). All the
                lists have the length of lidar numbers (5).
        """
        calibrations = sorted(
            frame.context.laser_calibrations, key=lambda c: c.name)
        points = []
        cp_points = []
        point_labels = []
        intensity = []
        elongation = []

        frame_pose = tf.convert_to_tensor(
            value=np.reshape(np.array(frame.pose.transform), [4, 4]))
        # [H, W, 6]
        range_image_top_pose_tensor = tf.reshape(
            tf.convert_to_tensor(value=range_image_top_pose.data),
            range_image_top_pose.shape.dims)
        # [H, W, 3, 3]
        range_image_top_pose_tensor_rotation = \
            transform_utils.get_rotation_matrix(
                range_image_top_pose_tensor[..., 0],
                range_image_top_pose_tensor[..., 1],
                range_image_top_pose_tensor[..., 2])
        range_image_top_pose_tensor_translation = \
            range_image_top_pose_tensor[..., 3:]
        range_image_top_pose_tensor = transform_utils.get_transform(
            range_image_top_pose_tensor_rotation,
            range_image_top_pose_tensor_translation)
        for c in calibrations:
            range_image = range_images[c.name][ri_index]
            if len(c.beam_inclinations) == 0:
                beam_inclinations = range_image_utils.compute_inclination(
                    tf.constant(
                        [c.beam_inclination_min, c.beam_inclination_max]),
                    height=range_image.shape.dims[0])
            else:
                beam_inclinations = tf.constant(c.beam_inclinations)

            beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
            extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

            range_image_tensor = tf.reshape(
                tf.convert_to_tensor(value=range_image.data),
                range_image.shape.dims)
            pixel_pose_local = None
            frame_pose_local = None
            if c.name == open_dataset.LaserName.TOP:
                pixel_pose_local = range_image_top_pose_tensor
                pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
                frame_pose_local = tf.expand_dims(frame_pose, axis=0)
            range_image_mask = range_image_tensor[..., 0] > 0

            if self.filter_no_label_zone_points:
                nlz_mask = range_image_tensor[..., 3] != 1.0  # 1.0: in NLZ
                range_image_mask = range_image_mask & nlz_mask

            range_image_cartesian = \
                range_image_utils.extract_point_cloud_from_range_image(
                    tf.expand_dims(range_image_tensor[..., 0], axis=0),
                    tf.expand_dims(extrinsic, axis=0),
                    tf.expand_dims(tf.convert_to_tensor(
                        value=beam_inclinations), axis=0),
                    pixel_pose=pixel_pose_local,
                    frame_pose=frame_pose_local)

            range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
            points_tensor = tf.gather_nd(range_image_cartesian,
                                         tf.compat.v1.where(range_image_mask))

            cp = camera_projections[c.name][ri_index]
            cp_tensor = tf.reshape(
                tf.convert_to_tensor(value=cp.data), cp.shape.dims)
            cp_points_tensor = tf.gather_nd(
                cp_tensor, tf.compat.v1.where(range_image_mask))
            points.append(points_tensor.numpy())
            cp_points.append(cp_points_tensor.numpy())

            if c.name in segmentation_labels:
                sl = segmentation_labels[c.name][ri_index]
                sl_tensor = tf.reshape(tf.convert_to_tensor(sl.data), sl.shape.dims)
                sl_points_tensor = tf.gather_nd(sl_tensor, tf.where(range_image_mask))
            else:
                sl_points_tensor = tf.zeros([points_tensor.shape[0], 2], dtype=tf.int32)

            point_labels.append(sl_points_tensor.numpy())

            intensity_tensor = tf.gather_nd(range_image_tensor[..., 1],
                                            tf.where(range_image_mask))
            intensity.append(intensity_tensor.numpy())

            elongation_tensor = tf.gather_nd(range_image_tensor[..., 2],
                                             tf.where(range_image_mask))
            elongation.append(elongation_tensor.numpy())

        return points, cp_points, point_labels, intensity, elongation

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
    converter = WaymoConverter('/nfs/s3_common_dataset/waymo_perception_v1.3/validation', \
                               '/nfs/volume-807-2/waymo_open_dataset_v_1_3_0/validation', 4)
    converter.convert()
