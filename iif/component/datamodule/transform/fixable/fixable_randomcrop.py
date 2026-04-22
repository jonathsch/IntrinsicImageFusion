from collections import defaultdict

import torchvision.transforms.functional as F
from torchvision.transforms import RandomCrop

from iif.utils.logging import init_logger
import torch


class FixableRandomCrop(RandomCrop):
    FIXED_PARAMS = defaultdict(lambda: None)

    def __init__(self,
                 size,
                 padding=None,
                 pad_if_needed=False,
                 fill=0,
                 padding_mode="constant",
                 center_only=False,
                 fixing_id: str = None):
        super().__init__(size, padding, pad_if_needed, fill, padding_mode)

        self.center_only = center_only
        self.fixing_id = fixing_id
        self.module_logger = init_logger()

    def reset_parameters(self):
        FixableRandomCrop.FIXED_PARAMS = defaultdict(lambda: None)

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        if self.center_only:
            return F.center_crop(img, self.size)

        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        _, height, width = F.get_dimensions(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)

        # Get the potentially fixed parameters
        if self.fixing_id is None:
            i, j, h, w = self.get_params(img, self.size)
        else:
            if FixableRandomCrop.FIXED_PARAMS[self.fixing_id] is None:
                FixableRandomCrop.FIXED_PARAMS[self.fixing_id] = self.get_params(img, self.size)
            i, j, h, w = FixableRandomCrop.FIXED_PARAMS[self.fixing_id]

        return F.crop(img, i, j, h, w)
