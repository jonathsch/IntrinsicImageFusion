from typing import Mapping, Callable

from torchvision import transforms as t

from iif.component.datamodule.transform.dynamic import DynamicTransform


class Compose(DynamicTransform):
    def __init__(self, transforms: Mapping[str, Callable], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        for t in self.transforms.values():
            if t is None:
                continue

            if isinstance(t, DynamicTransform):
                img = t(img, *args, **kwargs)
            else:
                img = t(img)
        return img
