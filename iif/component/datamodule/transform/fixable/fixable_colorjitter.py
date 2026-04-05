from collections import defaultdict
from typing import Union, Tuple, MutableMapping, Iterable

from torchvision.transforms import Compose, ColorJitter
import torchvision.transforms.functional as F

from iif.utils.logging import init_logger


class FixableColorJitter(ColorJitter):
    FIXED_PARAMS = defaultdict(lambda: None)

    def __init__(
        self,
        brightness: Union[float, Tuple[float, float]] = 0,
        contrast: Union[float, Tuple[float, float]] = 0,
        saturation: Union[float, Tuple[float, float]] = 0,
        hue: Union[float, Tuple[float, float]] = 0,
        fixing_id: str = None
    ) -> None:
        super().__init__(brightness=brightness,
                         contrast=contrast,
                         saturation=saturation,
                         hue=hue)
        self.fixing_id = fixing_id
        self.module_logger = init_logger()

    def reset_parameters(self):
        FixableColorJitter.FIXED_PARAMS = defaultdict(lambda: None)

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        if self.fixing_id is None:
            fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params(
                self.brightness, self.contrast, self.saturation, self.hue
            )
        else:
            if FixableColorJitter.FIXED_PARAMS[self.fixing_id] is None:
                FixableColorJitter.FIXED_PARAMS[self.fixing_id] = self.get_params(
                    self.brightness, self.contrast, self.saturation, self.hue
                )
            # else:
            #     self.module_logger.debug(f"Using fixed params for {self.__class__.__name__}.{self.fixing_id}!")
            fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = FixableColorJitter.FIXED_PARAMS[self.fixing_id]

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = F.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = F.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = F.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = F.adjust_hue(img, hue_factor)

        return img
