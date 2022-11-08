import torch

from mvseg3d.models.backbones import PointTransformer


if __name__ == '__main__':
    point_transformer = PointTransformer(16, 8)

    points = torch.randn((1800, 3)).cuda()
    features = torch.randn((1800, 16)).cuda()
    offsets = torch.IntTensor([900, 900]).cuda()
    out = point_transformer((points, features, offsets))
