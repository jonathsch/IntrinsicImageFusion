from typing import Union, List

import torch


class ZeroTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.zeros_like(x)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
