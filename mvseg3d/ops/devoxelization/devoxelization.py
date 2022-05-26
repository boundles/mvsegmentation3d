import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

from . import devoxelization_ext


class DevoxelizeFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half)
    def forward(ctx, feats: torch.Tensor,
                coords: torch.Tensor,
                weights: torch.Tensor) -> torch.Tensor:
        feats = feats.contiguous()
        coords = coords.contiguous().int()
        weights = weights.contiguous()

        if feats.device.type == 'cuda':
            output = devoxelization_ext.devoxelize_forward_cuda(
                feats, coords, weights)
        elif feats.device.type == 'cpu':
            output = devoxelization_ext.devoxelize_forward_cpu(
                feats, coords, weights)
        else:
            device = feats.device
            output = devoxelization_ext.devoxelize_forward_cpu(
                feats.cpu(), coords.cpu(), weights.cpu()).to(device)

        ctx.for_backwards = (coords, weights, feats.shape[0])
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: torch.Tensor):
        coords, weights, input_size = ctx.for_backwards
        grad_output = grad_output.contiguous()

        if grad_output.device.type == 'cuda':
            grad_feats = devoxelization_ext.devoxelize_backward_cuda(
                grad_output, coords, weights, input_size)
        elif grad_output.device.type == 'cpu':
            grad_feats = devoxelization_ext.devoxelize_backward_cpu(
                grad_output, coords, weights, input_size)
        else:
            device = grad_output.device
            grad_feats = devoxelization_ext.devoxelize_backward_cpu(
                grad_output.cpu(), coords.cpu(), weights.cpu(),
                input_size).to(device)

        return grad_feats, None, None

devoxelize = DevoxelizeFunction.apply