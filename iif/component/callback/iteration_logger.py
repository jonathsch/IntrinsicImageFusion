from typing import Any

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT

from iif.component.logger.base import log_anything
from iif.utils.logging import init_logger
import pytorch_lightning as pl


class IterationLogger(pl.Callback):
    def __init__(self,
                 context,
                 log_step=True,
                 log_epoch=True):
        super().__init__()
        self.module_logger = init_logger()

        self.context = f"{context}/" if context is not None else ""
        self.log_step = log_step
        self.log_epoch = log_epoch

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the train epoch begins."""
        if self.log_epoch:
            log_anything(logger=trainer.logger, name=f"{self.context}epoch", data=trainer.current_epoch)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the train epoch ends.

        To access all batch outputs at the end of the epoch, either:

        1. Implement `training_epoch_end` in the `LightningModule` and access outputs via the module OR
        2. Cache data across train batch hooks inside the callback implementation to post-process in this hook.
        """
        if self.log_epoch:
            log_anything(logger=trainer.logger, name=f"{self.context}epoch", data=trainer.current_epoch)

    def on_train_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int
    ) -> None:
        """Called when the train batch begins."""
        if self.log_step:
            log_anything(logger=trainer.logger, name=f"{self.context}step", data=trainer.global_step)

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        """Called when the train batch ends.

        Note:
            The value ``outputs["loss"]`` here will be the normalized value w.r.t ``accumulate_grad_batches`` of the
            loss returned from ``training_step``.

        """
        if self.log_step:
            log_anything(logger=trainer.logger, name=f"{self.context}step", data=trainer.global_step)
