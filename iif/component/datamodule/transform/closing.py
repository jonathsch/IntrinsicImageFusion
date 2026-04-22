from typing import Union, List

import torch
import torch.nn.functional as F
from kornia.morphology import dilation


class ClosingTransform(torch.nn.Module):
    """
    Simplified binary dilation transform.
    """
    def __init__(self, kernel_size=3, num_repeats=25):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_repeats = num_repeats

        self.dilate_kernel = torch.ones(self.kernel_size, self.kernel_size)
        self.dilate_structure = torch.zeros(self.kernel_size, self.kernel_size)

    def forward(self, x):
        for _ in range(25):
            x_dilated = dilation(x, kernel=self.dilate_kernel, structuring_element=self.dilate_structure)
            x = torch.where(x.mean(dim=1, keepdim=True) != 0, x, x_dilated)
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(kernel_size={self.kernel_size}, num_repeats={self.num_repeats})"
