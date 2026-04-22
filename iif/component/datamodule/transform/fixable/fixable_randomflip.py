from collections import defaultdict

import torchvision.transforms.functional as F
from iif.component.datamodule.transform.random_horizontal_flip import NormalRandomHorizontalFlip

NormalRandomHorizontalFlip
from iif.utils.logging import init_logger

import torch


class FixableRandomHorizontalFlip(NormalRandomHorizontalFlip):
    FIXED_PARAMS = defaultdict(lambda: None)

    def __init__(self, 
                 p=0.5,
                 flip_x=False,
                 fixing_id: str = None):
        super().__init__(p=p)
        
        self.fixing_id = fixing_id
        self.flip_x = flip_x
        self.module_logger = init_logger()

    def reset_parameters(self):
        FixableRandomHorizontalFlip.FIXED_PARAMS = defaultdict(lambda: None)

    def get_params(self, img):
        return torch.rand(1) < self.p

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        # Get the potentially fixed parameters
        if self.fixing_id is None:
            do_flip = self.get_params(img)
        else:
            if FixableRandomHorizontalFlip.FIXED_PARAMS[self.fixing_id] is None:
                FixableRandomHorizontalFlip.FIXED_PARAMS[self.fixing_id] = self.get_params(img)
            do_flip = FixableRandomHorizontalFlip.FIXED_PARAMS[self.fixing_id]

        if do_flip:
            img = F.hflip(img)
            if self.flip_x:
                img[..., 0, :, :] = -img[..., 0, :, :]
            return img
        return img
