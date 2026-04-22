from typing import Union, List

import torch


class MultiplyTransform(torch.nn.Module):
    def __init__(self, factor=1.):
        super().__init__()
        self.factor = torch.tensor(factor)

    def forward(self, x):
        return x * self.factor.to(x.device)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(factor={self.factor})"
