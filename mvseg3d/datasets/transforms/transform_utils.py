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

def random_flip_along_x(points):
    """
    Args:
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        points[:, 1] = -points[:, 1]

    return points

def random_flip_along_y(points):
    """
    Args:
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        points[:, 0] = -points[:, 0]

    return points

def random_translation_along_x(points, offset_std):
    """
    Args:
        points: (M, 3 + C),
        offset_std: float
    Returns:
    """
    offset = np.random.normal(0, offset_std, 1)
    points[:, 0] += offset
    return points

def random_translation_along_y(points, offset_std):
    """
    Args:
        points: (M, 3 + C),
        offset_std: float
    Returns:
    """
    offset = np.random.normal(0, offset_std, 1)
    points[:, 1] += offset
    return points

def random_translation_along_z(points, offset_std):
    """
    Args:
        points: (M, 3 + C),
        offset_std: float
    Returns:
    """
    offset = np.random.normal(0, offset_std, 1)
    points[:, 2] += offset
    return points