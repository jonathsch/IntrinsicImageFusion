# ------------------------------ Hydra Utils ------------------------------
import os

from iif.utils.config import parse_dynamic_cfg
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["HYDRA_FULL_ERROR"] = "1"  # Enable full Hydra error messages
os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "")

import sys

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

# ------------------------------ Custom Imports ------------------------------

from typing import Tuple

import pytorch_lightning as pl
import lovely_tensors as lt

from iif.utils.logging import init_logger

lt.monkey_patch()
module_logger = init_logger(os.path.basename(__file__))


def run_task(cfg: DictConfig):
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.seed is not None:
        pl.seed_everything(cfg.seed, workers=True)

    # Get the task
    task_cfg = cfg.task
    module_logger.info(f"Instantiating Task <{task_cfg._target_}>")
    task = hydra.utils.instantiate(task_cfg)

    # Log the task configuration
    task.log_config(task_cfg)

    # Run the task
    module_logger.info("Starting the task!")
    task.run()
    module_logger.info("Task finished!")


@hydra.main(version_base="1.3", config_path="../configs", config_name="job.yaml")
def main(cfg: DictConfig):
    # set the environment
    for ppath in os.environ["PYTHONPATH"].split(":"):
        module_logger.info(f"Adding {ppath} to path")
        sys.path.append(ppath)

    # Log the configuration
    OmegaConf.resolve(cfg)
    with open_dict(cfg):
        cfg = parse_dynamic_cfg(cfg)
    module_logger.info(f"Config: \n{OmegaConf.to_yaml(cfg)}")

    # Run the task
    try:
        run_task(cfg)
    except Exception as exc:
        module_logger.warning(exc)
        raise exc


if __name__ == "__main__":
    main()
