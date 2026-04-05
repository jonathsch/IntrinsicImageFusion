from typing import Optional, Dict

import torch
from pytorch_lightning.callbacks import LearningRateMonitor


class LearningRateChangeMonitor(LearningRateMonitor):
    def __init__(self,
                 context="lr",
                 logging_interval: Optional[str] = None,
                 log_momentum: bool = False) -> None:
        super().__init__(logging_interval, log_momentum)
        self.context = context
        self.prev_stat = dict()

    def _extract_stats(self, trainer: "pl.Trainer", interval: str) -> Dict[str, float]:
        latest_stat = {}

        (
            scheduler_hparam_keys,
            optimizers_with_scheduler,
            optimizers_with_scheduler_types,
        ) = self._find_names_from_schedulers(trainer.lr_scheduler_configs)
        self._remap_keys(scheduler_hparam_keys)

        for name, config in zip(scheduler_hparam_keys, trainer.lr_scheduler_configs):
            if interval in [config.interval, "any"]:
                opt = config.scheduler.optimizer
                current_stat = self._get_lr_momentum_stat(opt, name)
                latest_stat.update(current_stat)

        optimizer_hparam_keys, optimizers_without_scheduler = self._find_names_from_optimizers(
            trainer.optimizers,
            seen_optimizers=optimizers_with_scheduler,
            seen_optimizer_types=optimizers_with_scheduler_types,
        )
        self._remap_keys(optimizer_hparam_keys)

        for opt, names in zip(optimizers_without_scheduler, optimizer_hparam_keys):
            current_stat = self._get_lr_momentum_stat(opt, names)
            latest_stat.update(current_stat)

        latest_stat = self._filter_stats(latest_stat)

        latest_stat = self.add_context(latest_stat)

        trainer.callback_metrics.update(
            {name: torch.tensor(value, device=trainer.strategy.root_device) for name, value in latest_stat.items()}
        )

        return latest_stat

    def _filter_stats(self, latest_stat):
        filtered_stat = {}

        for key, value in latest_stat.items():
            if key not in self.prev_stat or self.prev_stat[key] != value:
                filtered_stat[key] = value
                self.prev_stat[key] = value
        return filtered_stat

    def add_context(self, stat):
        return {f"{self.context}/{key}": value for key, value in stat.items()}
