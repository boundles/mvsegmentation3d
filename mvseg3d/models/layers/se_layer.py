import torch.nn as nn


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Forward function.
        Args:
            x (torch.Tensor): The input shape (N, C) where C = dim of input
        Returns:
            torch.Tensor: The output with shape (N, C)
        """
        xin = x.unsqueeze(2).unsqueeze(3).permute(2, 1, 0, 3)
        b, c, _, _ = xin.size()
        out = self.avg_pool(xin).view(b, c)
        out = self.fc(out).view(b, c, 1, 1)
        out = xin * out.expand_as(xin)

        out = out.permute(2, 1, 0, 3).squeeze().squeeze()
        return out
