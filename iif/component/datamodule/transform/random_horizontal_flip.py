from typing import Union, List

from torchvision.transforms import RandomHorizontalFlip
import torchvision.transforms.functional as F
import torch


class NormalRandomHorizontalFlip(RandomHorizontalFlip):
    def forward(self, img):
        """
        Flips the normals accordingly
        """
        if torch.rand(1) < self.p:
            img = F.hflip(img)
            img[..., 0, :, :] = -img[..., 0, :, :]
            return img
        return img
