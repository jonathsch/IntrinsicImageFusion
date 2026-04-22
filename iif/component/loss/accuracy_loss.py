import torch
from torch import nn

from iif.utils.datastructure import Batch


class AccuracyLoss(nn.Module):
    def __init__(self,
                 **loss_kwargs):
        super().__init__()
        self.loss_kwargs = loss_kwargs

    def forward(self, preds, target):
        return (preds==target).float().mean()

