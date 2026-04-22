import torch

from iif.component.callback.scheduled_callback import ScheduledCallback
from iif.component.logger.base import log_anything
from iif.utils.attribute import rgetattr, rsetattr
from iif.utils.datastructure import Batch
from iif.utils.logging import init_logger


class ModelConfigChange(ScheduledCallback):    
    def __init__(self, 
                 change_type,
                 model_cfg,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.module_logger = init_logger()

        self.change_type = change_type
        self.model_cfg = Batch(model_cfg).flatten(separator=".")

    @torch.no_grad()
    def __call__(self, trainer, pl_module, outputs=Batch(), batch=Batch(), *args, **kwargs):
        self.module_logger.info(f"Changing the model config of {pl_module}")
        
        for key, value in self.model_cfg.items():
            old_val = rgetattr(pl_module, key)
            new_val = self.get_new_val(old_val, value)
            self.module_logger.info(f"Changing {key} from {old_val} to {new_val}")
            rsetattr(pl_module, key, new_val)

    def get_new_val(self, old_val, change_val):
        if self.change_type == 'multiply':
            return old_val * change_val
        elif self.change_type == 'add':
            return old_val + change_val
        elif self.change_type == 'set':
            return change_val
        else:
            raise NotImplementedError(f"Change type {self.change_type} not implemented.")

