import numpy as np
import torch
import torch.nn as nn


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionEncodingSine(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionEncodingSine, self).__init__()
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        """
        :param x: (N, 4), [batch_idx, z_idx, y_idx, x_idx]
        :return: Positional Encoding Matrix of size (N, channels)
        """
        pos_z = x[:, 1]
        pos_y = x[:, 2]
        pos_x = x[:, 3]
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb_y = get_emb(sin_inp_y)
        emb_z = get_emb(sin_inp_z)
        emb = torch.zeros((x.shape[0], self.channels * 3), device=x.device)
        emb[:, :self.channels] = emb_z
        emb[:, self.channels:2 * self.channels] = emb_y
        emb[:, 2 * self.channels:] = emb_x
        return emb
