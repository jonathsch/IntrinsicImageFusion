import torch
from torch import nn

from pbd.utils.datastructure import Batch


class CosineSimilarity(nn.Module):
    def __init__(self,
                 dim=1,
                 maximize=False):
        super().__init__()
        self.maximize = maximize
        self.similarity = torch.nn.CosineSimilarity(dim=dim)

    def forward(self, input, target):
        sim = self.similarity(input, target)

        if not self.maximize:
            sim = -sim

        return sim.mean()

