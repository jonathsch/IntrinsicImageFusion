from typing import Union, List

import torch


class ChromacityTransform(torch.nn.Module):
    def __init__(self, color_channel=1):
        super().__init__()
        self.color_channel = color_channel

    def forward(self, x):
        x_sum = torch.sum(x, dim=self.color_channel, keepdim=True)
        # x = torch.index_select(x, self.color_channel, torch.arange(0, x.shape[self.color_channel] - 1, device=x.device))
        x = x / (x_sum + 1e-6)
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
