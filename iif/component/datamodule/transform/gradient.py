from typing import Mapping, Callable

import torch
import torch.nn.functional as F
from torch import nn


class Gradient(torch.nn.Module):
    """
    Computes the square of the gradient of the input tensor.
    NOTE; Now supports only 2D
    """
    def __init__(self, abs_value=True):
        super().__init__()
        self.abs_value = abs_value
        self.sobel_x = nn.Parameter(torch.tensor([[1., 0., -1.],
                              [2, 0, -2],
                              [1, 0, -1]]).view((1, 1, 3, 3)), requires_grad=False)
        self.sobel_y = nn.Parameter(torch.tensor([[1., 2., 1.],
                          [0, 0, 0],
                          [-1, -2, -1]]).view((1, 1, 3, 3)), requires_grad=False)

    def forward(self, x) -> torch.Tensor:
        """
        Transforms the range of tensor.
        :param x: The input tensor
        :return: The transformed tensor
        """
        shape = x.shape
        self.to(x)
        x = x.view(-1, 1, *x.shape[-2:])
        x = F.pad(x, (1, 1, 1, 1), mode='reflect')
        G_x = F.conv2d(x, self.sobel_x)
        G_y = F.conv2d(x, self.sobel_y)
        if self.abs_value:
            G = G_x ** 2 + G_y ** 2
        else:
            G = G_x + G_y
        G = G.view(shape)
        return G

    def inverse(self, y):
        """
        Inverse transforms the range of tensor.
        :param y: The transformed tensor
        :return: The inverse transformed tensor
        """
        raise NotImplementedError(f"Gradient is not invertible")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
