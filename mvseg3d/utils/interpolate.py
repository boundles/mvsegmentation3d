import numpy as np

import torch
from torch import Tensor

def sparse_interpolate(input: Tensor,
                       input_coords: Tensor,
                       scale_factor: float,
                       output_coords: Tensor,
                       output_shape: tuple,
                       fill_value: int,
                       mode: str='nearest'):
    assert mode == 'nearest'
    h, w, l = output_shape
    input_index_dict = {}
    input_coords = torch.cat([input_coords[:, 0].unsqueeze(1), (input_coords[:, 1:] * scale_factor).int()], dim=1)
    input_index = input_coords[:, 0] * h * w * l + input_coords[:, 1] * w * l + input_coords[:, 2] * w + input_coords[:, 3]
    input_index = input_index.cpu().numpy()
    for i in range(input_index.shape[0]):
        idx = input_index[i]
        if not idx in input_index_dict:
            input_index_dict[idx] = input[i]

    output = np.full((output_coords.shape[0]), fill_value=fill_value)
    output_index = output_coords[:, 0] * h * w * l + output_coords[:, 1] * w * l + output_coords[:, 2] * w + output_coords[:, 3]
    output_index = output_index.cpu().numpy()
    for i in range(output_index.shape[0]):
        idx = output_index[i]
        if idx in input_index_dict:
            output[i] = input_index_dict[idx]
    output = torch.from_numpy(output).type(input.dtype).to(input.device)

    return output

if __name__ == '__main__':
    input = torch.Tensor([4, 5])
    input_coords = torch.Tensor([[0, 2, 2, 2], [1, 2, 2, 2]])
    scale_factor = 0.5
    output_coords = torch.Tensor([0, 1, 1, 1]).unsqueeze(0)
    output_shape = (10, 10, 10)
    output = sparse_interpolate(input, input_coords, scale_factor, output_coords, output_shape, 255)
    print(output)