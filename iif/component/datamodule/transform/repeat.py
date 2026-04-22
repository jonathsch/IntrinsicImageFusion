from typing import Union, List

import torch


class RepeatTransform(torch.nn.Module):
    def __init__(self,
                 repeats):
        super().__init__()
        self.repeats = repeats

    def forward(self, x):
        x = x.repeat(*self.repeats)
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
