import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

from . import voxel_pooling_ext

from torch_scatter import scatter_max


class VoxelAvgPoolingFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half)
    def forward(ctx, feats: torch.Tensor, coords: torch.Tensor,
                counts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ctx: context
            feats: FloatTensor[N, C]
            coords: the coordinates of points, IntTensor[N,]
            counts: point num of per voxel, IntTensor[M,]
        Returns:
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


class VoxelMaxPooling(object):
    def __call__(self, feats, data_dict):
        """
        Args:
            feats: FloatTensor[N, C]
            data_dict:
                point_voxel_ids: FloatTensor[N,]
        Returns:
            FloatTensor[M, C]
        """
        coords = data_dict['point_voxel_ids']
        mask = (coords != -1)
        feats = feats[mask]
        coords = coords[mask]
        data_dict['voxel_features'] = scatter_max(feats, coords, dim=0)
        return data_dict

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

voxel_max_pooling = VoxelMaxPooling()
