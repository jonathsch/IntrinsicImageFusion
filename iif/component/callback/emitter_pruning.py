import torch

from iif.component.callback.scheduled_callback import ScheduledCallback
from iif.component.logger.base import log_anything
from iif.component.model.emitter import SLFEmitter, SLFEnvmapImportanceEmitter
from iif.utils.attribute import rgetattr, rsetattr
from iif.utils.datastructure import Batch
from iif.utils.logging import init_logger


class EmitterPruning(ScheduledCallback):    
    def __init__(self, 
                 model,
                 pruning_cfg,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.module_logger = init_logger()
        self.pruning_cfg = pruning_cfg
        self.model = model

    @torch.no_grad()
    def __call__(self, trainer, pl_module, outputs=Batch(), batch=Batch(), *args, **kwargs):
        self.module_logger.info(f"Changing the model config of {pl_module}")
        
        model = rgetattr(pl_module, self.model)
        self.prune_emitters(model)

    def prune_emitters(self, model):
        assert isinstance(model, (SLFEmitter, SLFEnvmapImportanceEmitter)), f"EmitterPruning only supports SLFEmitter, got {type(model)}"
        model.prune_emitters(**self.pruning_cfg)
        