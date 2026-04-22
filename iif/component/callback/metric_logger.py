import hydra
import torch
from torchmetrics.image import PeakSignalNoiseRatio

from iif.component.callback.scheduled_callback import ScheduledCallback
from iif.component.logger.base import log_anything
from iif.utils.datastructure import Batch
from iif.utils.logging import init_logger


class MetricLogger(ScheduledCallback):
    def __init__(self,
                 loss_model_cfg=None,
                 context="metric",
                 is_metric=False,
                 log_schedule=None,
                 verbose=False):
        super().__init__(log_schedule=log_schedule, verbose=verbose)
        self.module_logger = init_logger()

        self.loss_model_cfg = loss_model_cfg
        self.loss_model = None

        self.context = context
        self.is_metric = is_metric

    def get_samples(self, outputs, batch):
        # Collect batch information
        loss_batch = Batch()
        loss_batch["input"] = batch
        loss_batch["output"] = outputs
        loss_batch.batchify()

        # Get the metrics
        output_samples = self.loss_model(loss_batch)
        output_samples = output_samples.map(lambda x: torch.atleast_1d(torch.tensor(x)))

        return output_samples

    @torch.no_grad()
    def __call__(self, trainer, pl_module, outputs=Batch(), batch=Batch(), *args, **kwargs):
        # Init within context
        # with trainer.init_module():
        self.loss_model = hydra.utils.instantiate(self.loss_model_cfg).to("cuda")  # TODO Upgrade PL 

        # Collect all related information
        samples = self.get_samples(outputs, batch)

        # Log the required data
        logged_data = log_anything(logger=trainer.logger, name=self.context, data=samples.map(list), is_metric=self.is_metric, step=trainer.fit_loop.epoch_loop._batches_that_stepped)

        # Log the metric
        if self.is_metric:
            trainer.callback_metrics.update(
                {name: torch.tensor(value, device=trainer.strategy.root_device) for name, value in logged_data.items()}
            )
