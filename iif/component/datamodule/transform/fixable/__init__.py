from typing import MutableMapping, Iterable

from torchvision.transforms import Compose


def reset_transform_params(transform):
    if isinstance(transform, MutableMapping):
        reset_transform_params(list(transform.values()))
    elif isinstance(transform, Iterable):
        for t in transform:
            reset_transform_params(t)
    elif isinstance(transform, Compose):
        reset_transform_params(transform.transforms)
    elif hasattr(transform, "reset_parameters"):
        transform.reset_parameters()