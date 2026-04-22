from pytorch_lightning.utilities import rank_zero_only

from .base import MyLogger
from iif.utils.logging import init_logger


class MultiLogger(MyLogger):
    def __init__(self, *loggers):
        super().__init__()
        self.loggers = list(loggers)
        self.module_logger = init_logger()
        self._active_logger = None

    def append(self, logger):
        self.loggers.append(logger)

    def get_checkpoint_path(self):
        raise NotImplementedError()

    def watch(self, model, log: str = "gradients", log_freq: int = 10, log_graph: bool = True):
        self.active_logger.watch(model, log, log_freq, log_graph)

    def log(self, data_dict):
        for logger in self.loggers:
            logger.log(data_dict)

    @property
    def active_logger(self):
        if self._active_logger is None:
            self._active_logger = self.loggers[0]
        return self._active_logger

    @property
    def experiment(self):
        return self.active_logger.experiment

    @property
    def name(self):
        return "MultiLogger"

    @property
    def save_dir(self):
        """Return the root directory where experiment logs get saved, or `None` if the logger does not save data
        locally."""
        return self.active_logger.save_dir

    @property
    def version(self):
        return self.active_logger.save_dir

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        return self.active_logger.log_hyperparams(params)

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        return self.active_logger.log_metrics(metrics, step)

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        return self.active_logger.save()

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        return self.active_logger.finalize(status)
