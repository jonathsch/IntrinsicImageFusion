import torch
from torch import nn
import torch.nn.functional as F

from iif.utils.image_io import save_image


class IntersectionOverUnion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input_mask = (input > 0.).any(dim=1)
        target_mask = (target > 0.).any(dim=1)

        intersection = (input_mask & target_mask).sum()
        union = (input_mask | target_mask).sum()
        
        if intersection == union:
            return torch.tensor(1.0, device=input.device)

        iou = intersection / union.clamp(min=1e-20)
        return iou
