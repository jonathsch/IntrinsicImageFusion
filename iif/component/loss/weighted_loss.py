import torch
from torch import nn

from pbd.utils.datastructure import Batch
from pbd.utils.image_io import show_image


class WeightedLoss(nn.Module):
    def __init__(self,
                 loss_layer: nn.Module,
                 **loss_kwargs):
        super().__init__()
        self.loss_layer = loss_layer
        self.loss_kwargs = loss_kwargs

    def forward(self, input, target, weight):
        return torch.mean(self.loss_layer(input=input, target=target, **self.loss_kwargs) * weight)

