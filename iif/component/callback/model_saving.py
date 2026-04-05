from typing import Any

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT

from iif.component.logger.base import log_anything
from iif.utils.attribute import rgetattr
from iif.utils.logging import init_logger
import pytorch_lightning as pl


class ModelSaving(pl.Callback):
    def __init__(self,
                 model_to_save: str,
                 out_path: str):
        super().__init__()
        self.module_logger = init_logger()

        self.model_to_save = model_to_save
        self.out_path = out_path

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the train ends."""
        self.module_logger.info("Training finished, saving the final model.")
        model = rgetattr(pl_module, self.model_to_save)
        
        if hasattr(model, 'save'):
            model.save(self.out_path)

        torch.save(model.state_dict(), self.out_path)
        self.module_logger.info(f"Model saved to {self.out_path}.")
