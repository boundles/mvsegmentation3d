import torch
import numpy as np

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False

def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:
    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa, sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot

class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.RandomGlobalScaling([0.95, 1.05]),
        >>>     transforms.RandomGlobalRotation([-0.78539816, 0.78539816]),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data_dict):
        for t in self.transforms:
            data_dict = t(data_dict)
        return data_dict

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string

class RandomGlobalScaling(object):
    def __init__(self, scale_range) -> None:
        self.scale_range = scale_range

    def __call__(self, data_dict):
        if self.scale_range[1] - self.scale_range[0] < 1e-3:
            return data_dict
        noise_scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        data_dict['points'][:, :3] *= noise_scale
        return data_dict

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RandomGlobalRotation(object):
    def __init__(self, rot_range) -> None:
        self.rot_range = rot_range

    def __call__(self, data_dict):
        noise_rotation = np.random.uniform(self.rot_range[0], self.rot_range[1])
        data_dict['points'] = rotate_points_along_z(data_dict['points'][np.newaxis, :, :], np.array([noise_rotation]))[0]

        return data_dict

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class PointShuffle(object):
    def __call__(self, data_dict):
        point_indices = np.array(range(data_dict['points'].shape[0]))
        np.random.shuffle(point_indices)

        data_dict['points'] = data_dict['points'][point_indices]

        cur_indices = data_dict.get('point_indices', None)
        point_image_features = data_dict.get('point_image_features', None)
        labels = data_dict.get('labels', None)

        if cur_indices is not None:
            cur_global_indices = []
            cur_local_indices = []
            for i, idx in enumerate(point_indices):
                # TODO: hard code here
                if idx <= cur_indices[-1]:
                    cur_global_indices.append(i)
                    cur_local_indices.append(idx)
            data_dict['point_indices'] = np.array(cur_global_indices)
            cur_indices = np.array(cur_local_indices)
        else:
            cur_indices = point_indices

        if point_image_features is not None:
            data_dict['point_image_features'] = point_image_features[cur_indices]

        if labels is not None:
            data_dict['labels'] = labels[cur_indices]

        return data_dict

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

if __name__ == '__main__':
    batch_dict = {'points': np.ones((100, 3))}
    transforms = Compose([RandomGlobalScaling([0.95, 1.05]),
                          RandomGlobalRotation([-0.78539816, 0.78539816]),
                          PointShuffle()])
    batch_dict = transforms(batch_dict)
    print(batch_dict)
