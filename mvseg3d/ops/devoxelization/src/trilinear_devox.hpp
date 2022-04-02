#ifndef TRILINEAR_DEVOX_HPP_
#define TRILINEAR_DEVOX_HPP_

#include <torch/torch.h>
#include <vector>

std::vector<at::Tensor> trilinear_devoxelize_forward(const int r,
                                                     const bool is_training,
                                                     const at::Tensor coords,
                                                     const at::Tensor features);

at::Tensor trilinear_devoxelize_backward(const at::Tensor grad_y,
                                         const at::Tensor indices,
                                         const at::Tensor weights, const int r);

#endif