import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

import torch_scatter

from . import voxel_pooling_ext


class VoxelAvgPoolingFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half)
    def forward(ctx, feats: torch.Tensor, coords: torch.Tensor,
                counts: torch.Tensor) -> torch.Tensor:
        """
            :param ctx:
            :param feats: FloatTensor[N, C]
            :param coords: the coordinates of points, IntTensor[N,]
            :param counts: point num of per voxel, IntTensor[M,]
            :return:
                FloatTensor[M, C]
        """
        feats = feats.contiguous()
        coords = coords.contiguous().int()
        counts = counts.int()

        if feats.device.type == 'cuda':
            output = voxel_pooling_ext.voxel_pooling_forward_cuda(
                feats, coords, counts)
        elif feats.device.type == 'cpu':
            output = voxel_pooling_ext.voxel_pooling_forward_cpu(
                feats, coords, counts)
        else:
            device = feats.device
            output = voxel_pooling_ext.voxel_pooling_forward_cpu(
                feats.cpu(), coords.cpu(), counts.cpu()).to(device)

        ctx.for_backwards = (coords, counts, feats.shape[0])
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: torch.Tensor):
        coords, counts, input_size = ctx.for_backwards
        grad_output = grad_output.contiguous()

        if grad_output.device.type == 'cuda':
            grad_feats = voxel_pooling_ext.voxel_pooling_backward_cuda(
                grad_output, coords, counts, input_size)
        elif grad_output.device.type == 'cpu':
            grad_feats = voxel_pooling_ext.voxel_pooling_backward_cpu(
                grad_output, coords, counts, input_size)
        else:
            device = grad_output.device
            grad_feats = voxel_pooling_ext.voxel_pooling_backward_cpu(
                grad_output.cpu(), coords.cpu(), counts.cpu(),
                input_size).to(device)

        return grad_feats, None, None

voxel_avg_pooling = VoxelAvgPoolingFunction.apply


def voxel_max_pooling(feats, coords):
    """
        :param feats: FloatTensor[N, C]
        :param coords: the coordinates of points, LongTensor[N,]
        :return:
            FloatTensor[M, C]
    """
    voxel_features = torch_scatter.scatter(feats, coords, dim=0, reduce='max')
    return voxel_features