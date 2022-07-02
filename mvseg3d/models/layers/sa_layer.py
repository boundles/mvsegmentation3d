import torch.nn as nn

from mvseg3d.utils.spconv_utils import replace_feature, conv_norm_act


class SALayer(nn.Module):
    def __init__(self, inplanes, planes, norm_fn, act_fn, indice_key):
        super(SALayer, self).__init__()
        self.conv = conv_norm_act(inplanes, planes, 3, norm_fn=norm_fn, act_fn=act_fn, padding=1, conv_type='subm',
                                  indice_key=indice_key)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward function.
        Args:
            x (SparseTensor): The input with features: shape (N, C)
        Returns:
            SparseTensor: The output with features: shape (N, C)
        """
        out = self.conv(x)
        out = replace_feature(out, self.sigmoid(out.features))
        out = replace_feature(out, x.features * out.features)

        return out
