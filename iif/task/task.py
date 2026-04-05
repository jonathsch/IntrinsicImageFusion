from abc import ABC
import os
import sys

from omegaconf import OmegaConf

from iif.utils.model import get_config


class Task(ABC):
    """
    Base class for tasks in the PIR framework.
    
    This class provides a structure for defining tasks, including methods for logging configuration
    and running the task. Subclasses should implement the `run` method to define specific task behavior.
    """
    TASK_NAME = "BaseTask"

    def log_config(self, cfg):
        """Log the configuration of the task."""
        # Log the overall config
        output_cfg = cfg.get("output", None)
        if output_cfg is not None:
            out_folder = output_cfg.get("folder_path", None)
            if out_folder is None:
                self.module_logger.warning("Output folder path not specified in config. Skipping config logging.")
            else:
                os.makedirs(out_folder, exist_ok=True)

                # Save the entire config
                cfg = get_config(cfg)
                OmegaConf.save(cfg, os.path.join(out_folder, f"config.yaml"))

                # Save the starting command
                with open(os.path.join(out_folder, "command.sh"), "w") as f:
                    modules = sys.argv[0].split("/")
                    command = f"python -m {'.'.join(modules[modules.index('priorbasedinverserendering'):-1])}.{modules[-1].replace('.py', '')} " + " ".join(sys.argv[1:])
                    f.write(command + "\n")

                self.module_logger.info(f"Config saved to {out_folder}")

    def run(self):
        """Run the task with the given configuration."""
        raise NotImplementedError("Subclasses must implement this method.")
