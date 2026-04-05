from typing import Mapping, Callable

import torch


class Normalize(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x) -> torch.Tensor:
        """
        Transforms the range of tensor.
        :param x: The input tensor
        :return: The transformed tensor
        """
        return torch.nn.functional.normalize(x, dim=self.dim, eps=1e-5)

    def inverse(self, y):
        """
        Inverse transforms the range of tensor.
        :param y: The transformed tensor
        :return: The inverse transformed tensor
        """
        raise NotImplementedError(f"Intensity normalization is not inversible")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dim={self.dim})"
