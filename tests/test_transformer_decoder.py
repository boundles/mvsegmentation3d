import torch

from mvseg3d.models.layers import MultiScaleTransformerDecoder


if __name__ == '__main__':
    decoder = MultiScaleTransformerDecoder(in_channels=[8, 4, 4], hidden_dim=4, num_queries=2, nheads=2,
                                           dim_feedforward=128, mask_dim=16, dec_layers=3)
    inputs = [torch.zeros((6, 8)), torch.zeros((12, 4)), torch.zeros((24, 4))]
    mask_features = torch.zeros((48, 16)).transpose(0, 1)
    out = decoder(inputs, mask_features)
    print(out.shape)

