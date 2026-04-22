from typing import Union, List

import torch


class ThresholdTransform(torch.nn.Module):
    def __init__(self, threshold=0.5, negate=False):
        super().__init__()
        self.negate = negate
        self.threshold = threshold

    def forward(self, x):
        dtype = x.dtype
        x = x > self.threshold
        if self.negate:
            x = ~x
        return x.to(dtype)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
