import os
from typing import Union, Optional

import wandb
from google.protobuf.internal.well_known_types import Any
from lightning_utilities.core.rank_zero import rank_zero_warn
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.logger import rank_zero_experiment
from wandb.sdk.lib import RunDisabled
from wandb.sdk.wandb_run import Run


class MyWandbLogger(WandbLogger):
    def __init__(
            self,
            name: Optional[str] = None,
            save_dir = ".",
            version: Optional[str] = None,
            offline: bool = False,
            dir: Optional[str] = None,
            id: Optional[str] = None,
            anonymous: Optional[bool] = None,
            project: str = "lightning_logs",
            log_model: Union[str, bool] = False,
            experiment: Union[Run, RunDisabled, None] = None,
            prefix: str = "",
            sweep_job_id: str = None,
            **kwargs: Any,
    ) -> None:
        slurm_array_job_id = ""
        if id is None:
            slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
            slurm_array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID", "")
            slurm_array_id = os.environ.get("SLURM_ARRAY_TASK_ID", "")
            if slurm_array_job_id == "":
                job_id = str(slurm_job_id)
            else:
                job_id = str(slurm_array_job_id)
                job_id += "_" + str(slurm_array_id)

            if job_id != "":
                id = f"{name}_{job_id}"

        if id is not None:
            id = id.replace("/", "-")

        if sweep_job_id is not None:
            kwargs["group"] = f"sweep_{name}_{slurm_array_job_id}"
            name = f"{name}_{sweep_job_id}"

        os.makedirs(save_dir, exist_ok=True)
        super().__init__(name=name,
                         save_dir=save_dir,
                         version=version,
                         offline=offline,
                         dir=dir,
                         id=id,
                         anonymous=anonymous,
                         project=project,
                         log_model=log_model,
                         experiment=experiment,
                         prefix=prefix,
                         **kwargs)

    @property
    @rank_zero_experiment
    def experiment(self) -> Union[Run, RunDisabled]:
        r"""

        Actual wandb object. To use wandb features in your
        :class:`~pytorch_lightning.core.module.LightningModule` do the following.

        Example::

        .. code-block:: python

            self.logger.experiment.some_wandb_function()

        """
        if self._experiment is None:
            if self._offline:
                os.environ["WANDB_MODE"] = "dryrun"

            attach_id = getattr(self, "_attach_id", None)
            if wandb.run is not None:
                # wandb process already created in this instance
                rank_zero_warn(
                    "There is a wandb run already in progress and newly created instances of `WandbLogger` will reuse"
                    " this run. If this is not desired, call `wandb.finish()` before instantiating `WandbLogger`."
                )
                self._experiment = wandb.run
            elif attach_id is not None and hasattr(wandb, "_attach"):
                # attach to wandb process referenced
                self._experiment = wandb._attach(attach_id)
            else:
                # create new wandb process
                self._experiment = wandb.init(**self._wandb_init, settings=wandb.Settings(start_method="fork"))

                # # define default x-axis
                # if isinstance(self._experiment, (Run, RunDisabled)) and getattr(
                #     self._experiment, "define_metric", None
                # ):
                #     self._experiment.define_metric("trainer/global_step")
                #     self._experiment.define_metric("*", step_metric="trainer/global_step", step_sync=True)

        assert isinstance(self._experiment, (Run, RunDisabled))
        return self._experiment
