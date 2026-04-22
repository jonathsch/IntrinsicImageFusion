import inspect
from abc import ABC
from collections import defaultdict
from typing import Any

import pytorch_lightning as pl
import wandb
from pytorch_lightning.utilities.types import STEP_OUTPUT

from iif.utils.distributed import get_local_rank


def none():
    return None


class ScheduledCallback(ABC, pl.Callback):
    def __init__(self, log_schedule=None, verbose=True, rank_zero_only=True):
        self.log_schedule = defaultdict(none)
        self.verbose = verbose
        self.rank_zero_only = rank_zero_only
        if log_schedule is not None:
            self.log_schedule.update(log_schedule)
        else:
            self.log_schedule["on_train_epoch_start"] = "::1"

        self.module_logger = None

        # Register the callbacks
        self.register_callbacks()

    def register_callbacks(self):
        def make_func(fn_name):
            return lambda caller, *args, **kwargs: caller.callback(fn_name, *args, **kwargs)

        for fn_name in [
            "setup",
            "teardown",
            "on_fit_start",
            "on_fit_end",
            "on_sanity_check_start",
            "on_sanity_check_end",
            "on_train_batch_start",
            "on_train_batch_end",
            "on_train_epoch_start",
            "on_train_epoch_end",
            "on_validation_epoch_start",
            "on_validation_epoch_end",
            "on_test_epoch_start",
            "on_test_epoch_end",
            "on_predict_epoch_start",
            "on_predict_epoch_end",
            "on_validation_batch_start",
            "on_validation_batch_end",
            "on_test_batch_start",
            "on_test_batch_end",
            "on_predict_batch_start",
            "on_predict_batch_end",
            "on_train_start",
            "on_train_end",
            "on_validation_start",
            "on_validation_end",
            "on_test_start",
            "on_test_end",
            "on_predict_start",
            "on_predict_end",
            "on_exception",
            "on_save_checkpoint",
            "on_load_checkpoint",
            "on_before_backward",
            "on_after_backward",
            "on_before_optimizer_step",
            "on_before_zero_grad"]:
            # Define dymaically only if it is not maunally overridden
            if getattr(self, fn_name).__code__ is getattr(pl.Callback, fn_name).__code__:
                setattr(ScheduledCallback, fn_name, make_func(fn_name))

    def callback(self, fn_name, trainer, *args, **kwargs):
        if not self.rank_zero_only or get_local_rank() <= 0:
            if self.should_log(fn_name, trainer):
                self.print_verbose(f"Starting callback {fn_name}")
                self(trainer, *args, **kwargs)
                self.print_verbose(f"Callback {fn_name} finished")

    def should_log(self, key, trainer):
        if trainer.logger is None:
            return False

        schedule = self.log_schedule[key]
        if schedule is None:
            return False

        if "epoch" in key:
            current_val = trainer.current_epoch
            max_val = trainer.max_epochs
        elif "batch" in key:
            current_val = trainer.global_step - 1
            max_val = None  # Not implemented
        elif "step" in key:
            current_val = trainer.global_step - 1
            max_val = None  # Not implemented
        else:
            raise NotImplementedError(f"Unknown key {key}")

        if isinstance(schedule, int):
            if schedule < 0:
                schedule = max_val - 1 - schedule - 1
            return current_val == schedule
        elif isinstance(schedule, list):
            return current_val in schedule
        elif isinstance(schedule, str):
            slice_def = schedule.split(":")

            start = 0
            if len(slice_def) >= 1:
                if slice_def[0] != "":
                    start = int(slice_def[0])

            end = None
            if len(slice_def) >= 2:
                if slice_def[1] != "":
                    end = int(slice_def[1])

            step = 1
            if len(slice_def) == 3:
                if slice_def[2] != "":
                    step = int(slice_def[2])

            if end is None:
                return current_val >= start and (current_val - start) % step == 0
            else:
                return current_val in range(start, end, step)

    def __call__(self, datamodule, logger, pl_module, outputs=None, batch=None, trainer=None):
        pass

    def log_metric(self, logger, value, ctx_name="vis", name="metric", name_prefix=""):
        logger.experiment.log({f"{ctx_name}/{name_prefix}{name}": value})

    def log_image(self, logger, image, ctx_name="vis", name="rerendered", name_prefix="", caption_prefix="",
                  caption=None):
        image = wandb.Image(image, caption=f"{caption_prefix}{'Rendered samples' if caption is None else caption}")
        logger.experiment.log({f"{ctx_name}/{name_prefix}{name}": image})

    def print_verbose(self, msg):
        if self.verbose:
            self.module_logger.info(msg)
