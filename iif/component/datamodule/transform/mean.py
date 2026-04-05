from typing import Union, List

import torch


class MeanTransform(torch.nn.Module):
    def __init__(self,
                 dim: Union[int, List[int]] = 1,
                 keepdim=True,
                 expand=True):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        self.expand = expand

    def forward(self, x):
        x_shape = x.shape
        x = torch.mean(x, dim=self.dim, keepdim=self.keepdim)
        if self.expand:
            x = x.expand(*x_shape)
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
