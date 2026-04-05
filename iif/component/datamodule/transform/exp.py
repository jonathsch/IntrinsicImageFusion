from typing import Union, List

import torch


class ExponentialTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.exp()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
