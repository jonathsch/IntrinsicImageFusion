from typing import Union, List

import torch


class AdditionTransform(torch.nn.Module):
    def __init__(self, bias=1.):
        super().__init__()
        self.bias = bias

    def forward(self, x):
        return x + self.bias

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(bias={self.bias})"
