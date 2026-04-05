import torch

from iif.component.callback.scheduled_callback import ScheduledCallback
from iif.component.logger.base import log_anything
from iif.utils.datastructure import Batch
from iif.utils.logging import init_logger


class ModelReinit(ScheduledCallback):    
    def __init__(self, reinit_args=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.module_logger = init_logger()
        self.reinit_args = reinit_args or dict()

    @torch.no_grad()
    def __call__(self, trainer, pl_module, outputs=Batch(), batch=Batch(), *args, **kwargs):
        self.module_logger.info(f"Reinitializing the model {pl_module}")
        pl_module.reinit(**self.reinit_args)
