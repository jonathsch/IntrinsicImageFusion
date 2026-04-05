import glob
import os
import torch
from tqdm import tqdm
import torchvision
from diffusers import DDIMScheduler
from iif.component.task.single_view_prediction.pipeline_rgb2x import StableDiffusionAOVMatEstPipeline
from iif.task.task import Task
from iif.utils.image_io import load_ldr_image
from iif.utils.logging import init_logger


class MultiViewPredictionAggregation(Task):
    """
    A task for single-view prediction in the PIR framework.
    
    This task is designed to handle single-view prediction scenarios, where the model predicts outputs based on a single input view.
    """
    TASK_NAME = "1_aggregate/rgbx"

    def __init__(self,
                 module,
                 datamodule,
                 trainer,
                 logger,
                 callback,
                 **kwargs):
        super().__init__()
        
        self.module = module
        self.datamodule = datamodule
        self.trainer = trainer
        self.logger = logger
        self.callback = callback

        self.module_logger = init_logger()

    def log_config(self, cfg):
        """Log the configuration of the single-view prediction task."""
        # Implement logging logic here
        if self.logger is not None:
            self.logger.log_hyperparams(cfg)

        if hasattr(self.module, 'log_config'):
            self.module.log_config(cfg)

    def run(self):
        # Run the optimization
        self.trainer.fit(model=self.module, datamodule=self.datamodule)
