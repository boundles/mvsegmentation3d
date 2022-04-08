#ifndef TRILINEAR_DEVOX_H_
#define TRILINEAR_DEVOX_H_

#include <torch/torch.h>

std::vector<at::Tensor> trilinear_devoxelize_forward(const int l, const int w, const int h,
                                                     const bool is_training,
                                                     const at::Tensor coords,
                                                     const at::Tensor features);

at::Tensor trilinear_devoxelize_backward(const at::Tensor grad_y,
                                         const at::Tensor indices,
                                         const at::Tensor weights,
                                         const int size);

// CUDA function declarations
void trilinear_devoxelize(int b, int c, int l, int w, int h, int n,
                          bool is_training, const float *coords,
                          const float *feat, int *inds, float *wgts,
                          float *outs);
void trilinear_devoxelize_grad(int b, int c, int size, int n, const int *inds,
                               const float *wgts, const float *grad_y,
                               float *grad_x);

#endif