from torch.autograd import Function

from . import devoxelization_ext


class TrilinearDevoxelization(Function):
    @staticmethod
    def forward(ctx, features, coords, is_training=True):
        """
        :param ctx:
        :param coords: the coordinates of points, FloatTensor[B, 3, N]
        :param features: FloatTensor[B, C, L, W, H]
        :param is_training: bool, training mode
        :return:
            FloatTensor[B, C, N]
        """
        B, C, L, W, H = features.shape
        features = features.contiguous().view(B, C, -1)
        coords = coords.contiguous()
        outs, inds, wgts = devoxelization_ext.trilinear_devoxelize_forward(L, W, H, is_training, coords, features)
        if is_training:
            ctx.save_for_backward(inds, wgts)
            ctx.l = L
            ctx.w = W
            ctx.h = H
        return outs

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx:
        :param grad_output: gradient of outputs, FloatTensor[B, C, N]
        :return:
            gradient of inputs, FloatTensor[B, C, L, W, H]
        """
        inds, wgts = ctx.saved_tensors
        grad_inputs = devoxelization_ext.trilinear_devoxelize_backward(grad_output.contiguous(), inds, wgts, ctx.l * ctx.w * ctx.h)
        return grad_inputs.view(grad_output.size(0), grad_output.size(1), ctx.l, ctx.w, ctx.h), None, None, None


trilinear_devoxelize = TrilinearDevoxelization.apply