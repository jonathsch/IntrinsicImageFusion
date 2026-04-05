from typing import Union, List

import torch


class GammaTransform(torch.nn.Module):
    def __init__(self, gamma=2.2, inverse=False):
        super().__init__()
        self.gamma = gamma if not inverse else 1 / gamma

    def forward(self, x):
        return torch.pow(x, 1 / self.gamma)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(gamma={self.gamma})"
