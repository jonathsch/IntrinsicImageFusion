from typing import Optional, Tuple, Type, Union, Dict, Callable

import torch
from torch.utils.data._utils.collate import default_collate_fn_map, collate

from .datastructure import Batch


def collate_namespace_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    elem = batch[0]
    return Batch(**{key: collate([getattr(d, key) for d in batch], collate_fn_map=collate_fn_map)
                                for key in vars(elem) if not key.startswith("_")})


def collate_dict_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    elem = batch[0]
    return {key: collate([d[key] for d in batch], collate_fn_map=collate_fn_map) for key in elem.keys()}


def collate_list_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    return torch.tensor(batch)


collate_fn_map = default_collate_fn_map.copy()
collate_fn_map[Batch] = collate_namespace_fn
collate_fn_map[list] = collate_list_fn
collate_fn_map[dict] = collate_dict_fn
