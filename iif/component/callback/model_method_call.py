import torch

from iif.component.callback.scheduled_callback import ScheduledCallback
from iif.component.logger.base import log_anything
from iif.utils.attribute import rgetattr, rsetattr
from iif.utils.datastructure import Batch
from iif.utils.logging import init_logger


class ModelMethodCall(ScheduledCallback):    
    def __init__(self, 
                 callbacks,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.module_logger = init_logger()

        self.callbacks = callbacks
        self.model_cfg = Batch(callbacks).flatten(separator=".")

    @torch.no_grad()
    def __call__(self, trainer, pl_module, outputs=Batch(), batch=Batch(), *args, **kwargs):        
        for key, value in self.callbacks.items():
            model = rgetattr(pl_module, key)
            self.module_logger.info(f"Calling {value}() of {value}")
            fn = getattr(model, value)
            fn()
