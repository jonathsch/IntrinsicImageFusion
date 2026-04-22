import torch

from iif.component.callback.scheduled_callback import ScheduledCallback
from iif.component.logger.base import log_anything
from iif.utils.datastructure import Batch
from iif.utils.logging import init_logger


class BatchLogger(ScheduledCallback):
    def __init__(self,
                 batch_keys_to_log=None,
                 output_keys_to_log=None,
                 context=None,
                 is_metric=False,
                 use_jpg=False,
                 transform=None,
                 log_schedule=None,
                 verbose=True):
        super().__init__(log_schedule=log_schedule, verbose=verbose)
        self.module_logger = init_logger()

        self.batch_keys_to_log = batch_keys_to_log
        self.output_keys_to_log = output_keys_to_log

        self.use_jpg = use_jpg

        self.context = context
        self.is_metric = is_metric

        self.transform = transform

    def get_samples(self, outputs, batch):
        # Collect batch information
        batch_samples = self._collect_samples(Batch(**batch), self.batch_keys_to_log)

        # Collect output information
        if isinstance(outputs, dict):
            output_samples = self._collect_samples(Batch(**outputs), self.output_keys_to_log)
        else:
            output_samples = Batch()

        return batch_samples.update(output_samples)

    def _collect_samples(self, batch, keys):
        samples = Batch()
        for key_to_log in keys:
            samples_to_log = batch.query_wildcard(key_to_log)
            samples_to_log = samples_to_log.map_keys(lambda x: x.replace('.', '/'))
            samples_to_log = samples_to_log.map(lambda x: torch.atleast_1d(x) if torch.is_tensor(x) else torch.atleast_1d(torch.tensor(x)))
            samples.update(samples_to_log)
        return samples

    @torch.no_grad()
    def __call__(self, trainer, pl_module, outputs=Batch(), batch=Batch(), *args, **kwargs):
        # Collect all related information
        samples = self.get_samples(outputs, batch)

        if self.transform is not None:
            samples = self.transform(samples)

        # Log the required data
        logged_data = log_anything(logger=trainer.logger, name=self.context, data=samples.map(list), is_metric=self.is_metric, step=trainer.fit_loop.epoch_loop._batches_that_stepped, use_jpg=self.use_jpg)

        # Log the metric
        if self.is_metric:
            trainer.callback_metrics.update(
                {name: torch.tensor(value, device=trainer.strategy.root_device) if not torch.is_tensor(value) else value.to(trainer.strategy.root_device)
                  for name, value in logged_data.items()}
            )
