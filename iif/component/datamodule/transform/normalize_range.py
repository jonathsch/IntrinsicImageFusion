from typing import Mapping, Callable

import torch


class NormalizeRange(torch.nn.Module):
    def __init__(self, input_range: list, output_range: list):
        super().__init__()
        self.input_range = input_range
        self.output_range = output_range
        assert self.output_range is not None, "Output range must be provided"

        if self.input_range is not None \
            and all((r is not None for r in self.input_range)):
            self.scale, self.shift = self.scale_shift(input_range=self.input_range, output_range=self.output_range)
        else:
            self.scale = None
            self.shift = None

    @staticmethod
    def scale_shift(input_range, output_range):
        scale = (output_range[1] - output_range[0]) / (input_range[1] - input_range[0] + 1e-6)
        shift = output_range[0] - input_range[0] * scale
        return scale, shift

    def forward(self, x) -> torch.Tensor:
        """
        Transforms the range of tensor.
        :param x: The input tensor
        :return: The transformed tensor
        """
        if self.scale is None:
            input_range = self.input_range
            if input_range is None:
                input_range = (x.min(), x.max())
            else:
                if input_range[0] is None:
                    input_range = (x.min(), input_range[1])
                else:
                    input_range = (input_range[0], x.max())
            scale, shift = self.scale_shift(input_range=input_range, output_range=self.output_range)
        else:
            scale = self.scale
            shift = self.shift
        return x * scale + shift

    def inverse(self, y):
        """
        Inverse transforms the range of tensor.
        :param y: The transformed tensor
        :return: The inverse transformed tensor
        """
        if self.scale is None:
            raise ValueError("Cannot inverse transform without input and output ranges")
        return (y - self.shift) / self.scale

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(input_range={self.input_range}, output_range={self.output_range})"
