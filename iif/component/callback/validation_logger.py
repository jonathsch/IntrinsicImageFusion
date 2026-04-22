import torch
from torch.utils.data import DataLoader

from iif.component.callback.scheduled_callback import ScheduledCallback
from iif.component.datamodule.sampler.subset_sequential import SubsetSequentialSampler
from iif.component.logger.base import log_anything
from iif.utils.config import range2list
from iif.utils.datastructure import Batch
from iif.utils.logging import init_logger
from iif.utils.stage import TrainStage


class ValidationLogger(ScheduledCallback):
    """
    Evaluates the model on specific samples and saves it to the output dictionary.
    """
    def __init__(self,
                 sample_ids,
                 forward_cfg=dict(),
                 stage=TrainStage.Training.value,
                 context="validation",
                 dataset_kwargs=dict(),
                 batch_size=1,
                 keys_to_average=None,
                 keys_to_log=None,
                 transform=None,
                 log_schedule=None,
                 verbose=True):
        super().__init__(log_schedule=log_schedule, verbose=verbose)
        self.module_logger = init_logger()

        self.dataset_kwargs = {("_target_" if k == "target" else k): v for k, v in dataset_kwargs.items()}
        self.forward_cfg = forward_cfg
        self.sample_ids = range2list(sample_ids)
        self.stage = stage
        self.context = context
        self.batch_size = batch_size
        self.transform = transform

        self.keys_to_average = keys_to_average
        self.keys_to_log = keys_to_log

        self._dataset = None

    def get_samples(self, datamodule, pl_module):
        # Collect all related information
        samples = Batch()
        samples.module_device = pl_module.device
        samples.scene_id = self.sample_ids

        if self._dataset is None:
            self._dataset = datamodule.load_dataset(self.stage, **self.dataset_kwargs)
        samples.dataset = self._dataset

        # Collect datapoints
        sampler = SubsetSequentialSampler(indices=self.sample_ids)
        samples.dataloader = DataLoader(samples.dataset, batch_size=self.batch_size, sampler=sampler, pin_memory=True)

        return samples

    @torch.no_grad()
    def __call__(self, trainer, pl_module, outputs, batch, *args, **kwargs):
        # Collect all related information
        samples = self.get_samples(trainer.datamodule, pl_module)

        # Iterate over the samples
        eval_outputs = []
        for batch in samples.dataloader:
            # Prepare the batch
            batch = trainer.precision_plugin.convert_input(batch)
            batch = trainer.lightning_module._on_before_batch_transfer(batch)
            batch = trainer.strategy.batch_to_device(batch)
            batch = Batch.from_dict(batch)

            with trainer.strategy.precision_plugin.train_step_context():
                # Run the model
                out = pl_module.validation(batch, **self.forward_cfg)
                eval_output = Batch(batch=batch, output=out)
                eval_outputs.append(eval_output.flatten(separator="/").query_wildcard(self.keys_to_log))
        
        eval_outputs = Batch.from_batch_list(*eval_outputs)
        eval_outputs = eval_outputs.map(torch.cat, dim=0)

        # Average the metrics
        if self.keys_to_average is not None:
            for key in self.keys_to_average:
                entries_to_average = eval_outputs.query_wildcard(self.keys_to_average)
                for key_to_average, entries in entries_to_average.items():
                    eval_outputs[key_to_average] = entries.mean(dim=0, keepdim=True)

        # Log the required data
        logged_data = log_anything(logger=trainer.logger, name=self.context, data=eval_outputs.map(list), step=trainer.fit_loop.epoch_loop._batches_that_stepped)
