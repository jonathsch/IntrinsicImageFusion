from typing import Union, List

import torch


class PowTransform(torch.nn.Module):
    def __init__(self, exponent=1.):
        super().__init__()
        self.exponent = exponent

    def forward(self, x):
        return x ** self.exponent

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(exponent={self.exponent})"
