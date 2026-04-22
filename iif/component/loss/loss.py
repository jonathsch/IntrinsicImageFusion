import torch
from torch import nn

from iif.utils.datastructure import Batch


class Loss(nn.Module):
    def __init__(self,
                 loss_layer: nn.Module,
                 data_keys: dict,
                 loss_arg_keys: list = None,
                 transform=None,
                 store_keys=None,
                 propagate_loss_info=False,
                 **loss_kwargs):
        super().__init__()
        self.loss_layer = loss_layer
        self.data_keys = data_keys
        self.loss_arg_keys = loss_arg_keys or list(data_keys.keys())
        self.transform = transform

        self.loss_kwargs = loss_kwargs
        self.store_keys = store_keys
        self.propagate_loss_info = propagate_loss_info

    def forward(self, batch, loss_info=None):
        # Collect the inputs
        data = Batch({name: batch[key] for name, key in self.data_keys.items()})
        data = data.map_keys(lambda x: x.replace('.', '/'))

        # Apply the transform
        if self.transform is not None:
            data = self.transform(Batch(**data))

        # Save images if requested
        if self.store_keys is not None and loss_info is not None:
            extra_info = Batch(**{name: data[key] for name, key in self.store_keys.items()})
            if "extra_loss_data" in loss_info:
                loss_info["extra_loss_data"].update(extra_info)
            else:
                loss_info["extra_loss_data"] = extra_info

        # Move loss layer to device if possible
        loss_layer = self.loss_layer
        if hasattr(batch, "to"):
            loss_layer = loss_layer.to(list(data.flatten().values())[0].device)

        if self.propagate_loss_info:
            return loss_layer(**data[self.loss_arg_keys], **self.loss_kwargs, loss_info=loss_info)
        else:
            return loss_layer(**data[self.loss_arg_keys], **self.loss_kwargs)


class ComposedLoss(nn.Module):
    def __init__(self,
                 losses: dict):
        super().__init__()
        self.losses = nn.ModuleDict({loss_name: loss_params["loss"] for loss_name, loss_params in losses.items() if loss_params is not None})
        self.loss_weights = {loss_name: loss_params["weight"] if "weight" in loss_params else 1 for loss_name, loss_params in losses.items() if loss_params is not None}

    def forward(self, batch):
        loss_info = Batch()
        total_loss = 0
        for loss_name, loss in self.losses.items():
            weight = self.loss_weights[loss_name]
            if weight is 0:
                loss_val = 0
            else:
                loss_val = weight * loss(batch, loss_info=loss_info)

            loss_info[loss_name] = loss_val

            if isinstance(loss_val, Batch):
                total_loss += loss_val["loss"]
            else:
                total_loss += loss_val

        loss_info["loss"] = total_loss
        loss_info = loss_info.flatten(separator='_')
        return loss_info
