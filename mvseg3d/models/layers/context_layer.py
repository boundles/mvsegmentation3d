import torch
import torch.nn as nn

from torch_scatter import scatter

from mvseg3d.utils.spconv_utils import replace_feature, ConvModule


class PointPooling(nn.Module):
    def __init__(self, in_channels, channels):
        super(PointPooling, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, channels, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        indices = x.indices[:, 0].long()
        out = scatter(x.features, indices, dim=0, reduce='mean')
        out = self.fc(out)
        out = out[indices]
        return out


class ASPPModule(nn.ModuleList):
    """Atrous Spatial Pyramid Pooling (ASPP) Module.

    Args:
        dilations (tuple[int]): Dilation rate of each layer.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        norm_fn: Norm layer.
        act_fn: Activation layer.
    """

    def __init__(self, dilations, in_channels, channels, norm_fn, act_fn, indice_key):
        super(ASPPModule, self).__init__()
        self.dilations = dilations
        self.in_channels = in_channels
        self.channels = channels
        self.norm_fn = norm_fn
        self.act_fn = act_fn
        self.aspp_modules = []
        for dilation in dilations:
            self.aspp_modules.append(
                ConvModule(
                    self.in_channels,
                    self.channels,
                    1 if dilation == 1 else 3,
                    dilation=dilation,
                    padding=0 if dilation == 1 else dilation,
                    norm_fn=norm_fn,
                    act_fn=act_fn,
                    indice_key=indice_key + str(dilation)))
        self.aspp_modules = nn.ModuleList(self.aspp_modules)

    def forward(self, x):
        """Forward function."""
        aspp_outs = []

        for aspp_module in self.aspp_modules:
            aspp_out = aspp_module(x)
            aspp_outs.append(aspp_out)

        return aspp_outs


class ContextLayer(nn.Module):
    def __init__(self, dilations, in_channels, channels, act_fn, norm_fn, indice_key):
        super(ContextLayer, self).__init__()
        self.aspp_modules = ASPPModule(dilations, in_channels, channels, norm_fn=norm_fn,
                                       act_fn=act_fn, indice_key=indice_key)
        self.bottleneck = nn.Sequential(
            nn.Linear(len(dilations) * channels, in_channels, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """Forward function.
        Args:
            x (SparseTensor): The input with features: shape (N, C)
        Returns:
            Features with context information
        """
        aspp_outs = [aspp_out.features for aspp_out in self.aspp_modules(x)]
        aspp_outs = torch.cat(aspp_outs, dim=1)
        aspp_outs = self.bottleneck(aspp_outs)
        x = replace_feature(x, aspp_outs)
        return x
