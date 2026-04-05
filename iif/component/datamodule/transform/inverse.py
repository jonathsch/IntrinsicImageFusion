from typing import Union, List

import torch


class InverseTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = 1 / (x + 1e-6)
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
