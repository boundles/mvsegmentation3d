#pragma once

#include <vector>

#include <pybind11/pybind11.h>

#include <torch/torch.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>


at::Tensor devoxelize_forward_cpu(const at::Tensor feat,
                                  const at::Tensor indices,
                                  const at::Tensor weight);

at::Tensor devoxelize_backward_cpu(const at::Tensor top_grad,
                                   const at::Tensor indices,
                                   const at::Tensor weight, int n);

at::Tensor devoxelize_forward_cuda(const at::Tensor feat,
                                   const at::Tensor indices,
                                   const at::Tensor weight);

at::Tensor devoxelize_backward_cuda(const at::Tensor top_grad,
                                    const at::Tensor indices,
                                    const at::Tensor weight, int n);