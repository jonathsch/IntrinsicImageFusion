from typing import Mapping, Callable

import torch


class NormalizeIntensity(torch.nn.Module):
    def __init__(self, output_mean):
        super().__init__()
        self.output_mean = output_mean

    def forward(self, x) -> torch.Tensor:
        """
        Transforms the range of tensor.
        :param x: The input tensor
        :return: The transformed tensor
        """
        return x / (x.mean() + 1e-6) * self.output_mean

    def inverse(self, y):
        """
        Inverse transforms the range of tensor.
        :param y: The transformed tensor
        :return: The inverse transformed tensor
        """
        raise NotImplementedError(f"Intensity normalization is not invertible")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(output_mean={self.output_mean})"
