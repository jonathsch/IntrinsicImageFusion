from typing import Union, List

import torch
import torch.nn.functional as F
from kornia.morphology import erosion


class ErosionTransform(torch.nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size

        self.kernel = torch.ones(self.kernel_size, self.kernel_size)

    def forward(self, x):
        x = erosion(x, self.kernel)
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
