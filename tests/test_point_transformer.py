import torch

from mvseg3d.models.backbones import PointTransformer


if __name__ == '__main__':
    model = PointTransformer(16, 8)
    model = model.cuda()

    points = torch.randn((1800, 3)).cuda()
    features = torch.randn((1800, 16)).cuda()
    offsets = torch.IntTensor([900, 1800]).cuda()
    out = model((points, features, offsets))
    print(out.shape)
