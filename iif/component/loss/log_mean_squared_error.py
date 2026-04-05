import torch
from torch import nn
import torch.nn.functional as F


class LogMeanSquaredError(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return F.mse_loss(torch.log(input+1), torch.log(target+1))
