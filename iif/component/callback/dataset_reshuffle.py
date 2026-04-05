import torch

from iif.component.callback.scheduled_callback import ScheduledCallback
from iif.component.logger.base import log_anything
from iif.utils.datastructure import Batch
from iif.utils.logging import init_logger


class DatasetReshuffle(ScheduledCallback):    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.module_logger = init_logger()

    @torch.no_grad()
    def __call__(self, trainer, pl_module, outputs=Batch(), batch=Batch(), *args, **kwargs):
        self.module_logger.info(f"Reshuffling dataset {trainer.datamodule.dataset_train}")
        trainer.datamodule.dataset_train.reshuffle()
