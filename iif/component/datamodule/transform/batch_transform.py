import copy
from typing import Mapping, Callable, Union, MutableMapping

import torch

from iif.utils.datastructure import Batch
from .dynamic import DynamicTransform
from .fixable import reset_transform_params


class BatchTransform(torch.nn.Module):
    def __init__(self, transform: Union[Mapping[str, Callable], Callable], reset_params=False, *args, **kwargs):
        super().__init__()
        self.transform = transform
        self.reset_params = reset_params

    def __getitem__(self, index) -> Callable:
        if isinstance(self.transform, Mapping):
            return self.transform.get(index, self.transform.get("_default", None))
        else:
            return self.transform

    def forward(self, x_dict: Batch) -> Mapping[str, torch.Tensor]:
        """
        Transforms the elements of a dictionary according to the transform table.
        :param x_dict: The input dictionary
        :return: The transformed dictionary
        """
        x_out = x_dict
        if self.reset_params:
            self.reset_parameters()
        for key in x_dict.keys(recursive=True):
            transform = self[key]
            if transform is not None:
                val = x_dict[key]

                if isinstance(transform, MutableMapping):
                    for out_key, t in transform.items():
                        x_out[out_key] = self.eval_transform(t, val, x_dict)
                else:
                    x_out[key] = self.eval_transform(transform, val, x_dict)

        return x_out

    def eval_transform(self, transform, val, batch):
        if isinstance(transform, DynamicTransform):
            return transform(val, batch)
        else:
            return transform(val)

    def inverse(self, x_trans_dict: Batch) -> Mapping[str, torch.Tensor]:
        """
        Inverse transforms the elements of a dictionary according to the transform table.
        :param x_dict: The transformed dictionary
        :return: The inverse transformed dictionary
        """
        x_out = x_trans_dict
        for key in x_trans_dict.keys(recursive=True):
            val = x_trans_dict[key]
            if self[key] is not None and hasattr(self[key], "inverse"):
                x_out[key] = self[key].inverse(val)
            else:
                x_out[key] = val

        return x_out

    def reset_parameters(self):
        reset_transform_params(self.transform)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(transform={self.transform})"


class GetKeyTransform(torch.nn.Module):
    def __init__(self,
                 key):
        super().__init__()
        self.key = key

    def forward(self, x):
        return x[self.key]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(key={self.key})"


class AggregateTransform(torch.nn.Module):
    def __init__(self,
                 keys_to_aggregate,
                 aggregate_fn):
        super().__init__()
        self.keys_to_aggregate = keys_to_aggregate
        self.aggregate_fn = aggregate_fn

    def forward(self, x):
        # TODO
        out = 1
        for key in self.keys_to_aggregate:
            out *= x[key]
        return out
        # return self.aggregate_fn(*x[self.keys_to_aggregate].values())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(key={self.keys_to_aggregate}, aggregate_fn={self.aggregate_fn})"
