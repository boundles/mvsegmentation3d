import numpy as np


def swap(points1, point_image_features1, labels1, points2, point_image_features2, labels2, start_angle, end_angle):
    # calculate horizontal angle for each point
    yaw1 = -np.arctan2(points1[:, 1], points1[:, 0])
    yaw2 = -np.arctan2(points2[:, 1], points2[:, 0])

    # select points in sector
    indices1 = np.where((yaw1 > start_angle) & (yaw1 < end_angle))
    indices2 = np.where((yaw2 > start_angle) & (yaw2 < end_angle))

    # swap
    points1_out = np.delete(points1, indices1, axis=0)
    points1_out = np.concatenate((points1_out, points2[indices2]))
    point_image_features1_out = np.delete(point_image_features1, indices1, axis=0)
    point_image_features1_out = np.concatenate((point_image_features1_out, point_image_features2[indices2]))
    labels1_out = np.delete(labels1, indices1)
    labels1_out = np.concatenate((labels1_out, labels2[indices2]))

    return points1_out, point_image_features1_out, labels1_out


def rotate_copy(points, point_image_features, labels, instance_classes, rot_angle_range):
    # extract instance points
    points_inst, labels_inst, point_image_features_inst = [], [], []
    for s_class in instance_classes:
        point_indices = np.where((labels == s_class))
        points_inst.append(points[point_indices])
        point_image_features_inst.append(point_image_features[point_indices])
        labels_inst.append(labels[point_indices])
    points_inst = np.concatenate(points_inst, axis=0)
    point_image_features_inst = np.concatenate(point_image_features_inst, axis=0)
    labels_inst = np.concatenate(labels_inst, axis=0)

    # rotate-copy
    points_copy = [points_inst]
    point_image_features_copy = [point_image_features_inst]
    labels_copy = [labels_inst]
    for angle in rot_angle_range:
        rot_mat = np.array([[np.cos(angle),
                             np.sin(angle), 0],
                            [-np.sin(angle),
                             np.cos(angle), 0], [0, 0, 1]])
        new_point = np.zeros_like(points_inst)
        new_point[:, :3] = np.dot(points_inst[:, :3], rot_mat)
        new_point[:, 3:] = points_inst[:, 3:]
        points_copy.append(new_point)
        point_image_features_copy.append(point_image_features_inst)
        labels_copy.append(labels_inst)
    points_copy = np.concatenate(points_copy, axis=0)
    point_image_features_copy = np.concatenate(point_image_features_copy, axis=0)
    labels_copy = np.concatenate(labels_copy, axis=0)
    return points_copy, point_image_features_copy, labels_copy


class PolarMix(object):
    def __init__(self, instance_classes, rot_angle_range):
        self.instance_classes = instance_classes
        self.rot_angle_range = rot_angle_range

    def __call__(self, points1, point_image_features1, labels1, points2, point_image_features2, labels2, alpha, beta):
        points_out, point_image_features_out, labels_out = points1, point_image_features1, labels1

        # swapping
        if np.random.random() < 0.5:
            points_out, point_image_features_out, labels_out = swap(points1, point_image_features1, labels1,
                                                                    points2, point_image_features2, labels2,
                                                                    start_angle=alpha, end_angle=beta)

        # rotate-pasting
        if np.random.random() < 1.0:
            # rotate-copy
            points_copy, point_image_features_copy, labels_copy = rotate_copy(points2, point_image_features2, labels2,
                                                                              self.instance_classes, self.rot_angle_range)
            # paste
            points_out = np.concatenate((points_out, points_copy), axis=0)
            point_image_features_out = np.concatenate((point_image_features_out, point_image_features_copy), axis=0)
            labels_out = np.concatenate((labels_out, labels_copy), axis=0)

        return points_out, point_image_features_out, labels_out
