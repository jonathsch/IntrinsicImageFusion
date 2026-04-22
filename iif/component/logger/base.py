from abc import abstractmethod

import torch
import wandb
from torchvision.transforms import ToPILImage
import PIL

from iif.utils.datastructure import Batch
from iif.utils.logging import init_logger

module_logger = init_logger("LOGGING")


try:
    from pytorch_lightning.loggers import Logger

    class MyLogger(Logger):
        def get_checkpoint_path(self):
            pass
    
        @property
        def experiment(self):
            return self
    
        @abstractmethod
        def log(self, data_dict):
            pass
    
        def watch(self, model, log: str = "gradients", log_freq: int = 10, log_graph: bool = True):
            pass
except ImportError:
    pass


def anything_to_wandb(name, data, use_jpg=True, **kwargs):
    to_pil = ToPILImage()
    kwargs = Batch(**kwargs)
    if isinstance(data, torch.Tensor):
        if data.dtype == torch.int64:
            module_logger.warning(f"Logging {name} of type int64 is not implemented!")
            data = None
        else:
            if data.ndim == 0:  # Single scalar           - Logged as point sample
                pass
            elif data.ndim == 1:  # Single vector         - Logged as histogram
                data = wandb.plot.histogram(wandb.Table(data=[[s] for s in data.cpu()], columns=["values"]), "values")
            elif data.ndim == 2:  # Single-channel image  - Logged as image
                data = wandb.Image(to_pil(data.unsqueeze(0).clamp(0, 1)), caption=kwargs.get('caption', None), file_type='jpeg' if use_jpg else 'png')
            elif data.ndim == 3:  # Multi-channel image   - Logged as image
                if data.shape[0] == 1 or data.shape[0] == 3:
                    data = wandb.Image(to_pil(data.clamp(0, 1)), caption=kwargs.get('caption', None), file_type='jpeg' if use_jpg else 'png')
                else:
                    module_logger.warning(f"Logging {name} in shape of {data.shape} is not implemented!")
                    data = None
            elif data.ndim == 4:  # Video                 - Logged as video
                data = wandb.Video((data.clamp(0, 1).cpu() * 255).to(torch.uint8),
                                caption=kwargs.get('caption', None), format="mp4", fps=kwargs.get('fps', 4))
            else:
                module_logger.warning(f"Logging {name} in shape of {data.shape} is not implemented!")
                data = None
        data = {name: data}
    elif isinstance(data, PIL.Image.Image):
        data = {name: wandb.Image(data, caption=kwargs.get('caption', None), file_type='jpeg' if use_jpg else 'png')}
    elif isinstance(data, (list, tuple)):
        kwargs = kwargs.to_list()
        if len(kwargs) == 0:
            kwargs = [dict() for _ in range(len(data))]
        prepared_data = [anything_to_wandb(name, d, **kwa)[name] for d, kwa in zip(data, kwargs)]
        if len(prepared_data) == 1:  # Avoid logging scalar as histogram
            prepared_data = prepared_data[0]
        data = {name: prepared_data}
    elif isinstance(data, (dict, Batch)):
        prepared_data = Batch()
        name_prefix = f"{name}/" if name is not None else ""
        for key, value in data.items():
            prepared_data.update(**anything_to_wandb(f"{name_prefix}{key}", value, **kwargs))
        data = prepared_data.flatten(separator="/").to_dict()
    elif isinstance(data, (int, float)):
        data = {name: data}
    elif isinstance(data, (wandb.Image, wandb.Table, wandb.Video)):
        data = {name: data}
    else:
        module_logger.warning(f"Logging {name} of type {type(data)} is not implemented!")
        data = {name: None}

    return data


def log_anything(logger, name, data, is_metric=False, step=None, use_jpg=False, **kwargs):
    """
    Generic method to log anything
    :param logger:
    :param data:
    :param caption:
    :return:
    """
    data = anything_to_wandb(name, data, use_jpg=use_jpg, **kwargs)
    if is_metric:
        logger.log_metrics(data, step=step)
    else:
        logger.experiment.log(data)

    return data
