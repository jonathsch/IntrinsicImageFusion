from abc import ABC, abstractmethod
from typing import Any

import hydra
import torch
from pytorch_lightning import LightningModule

from iif.utils.datastructure import Batch
from iif.utils.logging import init_logger
from iif.utils.stage import TrainStage


class TrainableModule(LightningModule, ABC):
    def __init__(self,
                 loss_model_cfg=None,
                 optimizer_cfg=None,
                 scheduler_cfg=None,
                 ckpt=None,
                 *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.module_logger = init_logger()

        self.ckpt = ckpt

        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

        self.loss_model_cfg = loss_model_cfg
        self.loss_model = None

    def configure_model(self):
        self.loss_model = hydra.utils.instantiate(self.loss_model_cfg)
        self.optimizer_cfg = hydra.utils.instantiate(self.optimizer_cfg)
        self.scheduler_cfg = hydra.utils.instantiate(self.scheduler_cfg)

        if self.ckpt is not None:
            self.init_from_ckpt(self.ckpt)

    def init_from_ckpt(self, path):
        """
        Initialize the model from a checkpoint
        :param path: Checkpoint path
        """
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]

        missing, unexpected = self.load_state_dict(sd, strict=False)
        self.module_logger.info(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            self.module_logger.warning(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            self.module_logger.warning(f"Unexpected Keys: {unexpected}")

    def general_step(self, batch, batch_idx, mode: TrainStage):
        """
        General step used in all phases
        :param batch: The current batch of data
        :param batch_idx: The current batch index
        :param mode: The current phase
        :return: The loss
        """
        # ======================== STEP ========================
        # Run the model
        output = self.step(batch)

        # Calculate loss
        loss_info = self.calculate_loss(batch, output)

        output.update(loss_info)
        return output

    @abstractmethod
    def step(self, batch):
        pass

    def calculate_loss(self, batch, output):
        loss_batch = Batch()
        loss_batch["input"] = batch
        loss_batch["output"] = output

        return self.loss_model(loss_batch)

    def on_after_backward(self) -> None:
        pass
        # for name, param in self.named_parameters():
        #     if param.grad is None and param.requires_grad:
        #         self.module_logger.warning(f"Untrainable parameter found: {name}")

    def training_step(self, batch, batch_idx, *args):
        """Abstract definition of the training step"""
        return self.general_step(batch, batch_idx, TrainStage.Training)

    def validation_step(self, batch, batch_idx):
        """Abstract definition of the validation step"""
        return self.general_step(batch, batch_idx, TrainStage.Validation)

    def configure_optimizers(self):
        optimizer = self.optimizer_cfg(params=self.parameters())
        if self.scheduler_cfg is not None:
            scheduler = self.scheduler_cfg(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "loss/train_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
