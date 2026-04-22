from typing import Union, List

import torch
import torch.nn.functional as F


class DilationTransform(torch.nn.Module):
    """
    Simplified binary dilation transform.
    """
    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size

        self.dilate = torch.nn.Parameter(torch.ones(1, 1, self.kernel_size, self.kernel_size),
                                          requires_grad=False)

    def forward(self, x):
        shape = x.shape
        x = x.view(-1, 1, *x.shape[-2:])
        x = F.pad(x, (self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2), mode='reflect')
        x = F.conv2d(x, self.dilate)
        x = torch.clamp(x, 0, 1)
        x = x.view(shape)
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
