import torch
from torch import nn

from pbd.utils.datastructure import Batch


class ScaleInvariantLoss(nn.Module):
    def __init__(self,
                 loss_layer,
                 invariant_channels=None,
                 **loss_kwargs):
        super().__init__()
        self.loss_layer = loss_layer
        self.loss_kwargs = loss_kwargs

        self.invariant_channels = invariant_channels

    def forward(self, input, target):
        # Scale the input to the target
        target_mean = target.mean(dim=self.invariant_channels, keepdim=True)
        input_mean = input.mean(dim=self.invariant_channels, keepdim=True)
        input = input / input_mean * target_mean

        return self.loss_layer(input, target, **self.loss_kwargs)

