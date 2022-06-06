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
        x = x.unsqueeze(2).unsqueeze(3).permute(2, 1, 0, 3)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = x * y.expand_as(x)

        x = x.permute(2, 1, 0, 3).squeeze().squeeze()
        y = y.permute(2, 1, 0, 3).squeeze().squeeze()
        return x + y