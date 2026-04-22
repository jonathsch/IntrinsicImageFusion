import os

import cv2
import numpy as np
import wandb
from pytorch_lightning.utilities import rank_zero_only

from .base import MyLogger
from iif.utils.image_io import save_exr_image
from iif.utils.logging import init_logger


class ConsoleLogger(MyLogger):
    def __init__(self,
                 name=None,
                 id=None,
                 save_dir=None,
                 project=None,
                 entity=None,
                 plot_images=True,
                 save_images=False,
                 log_folder=None,
                 save_HDR=False,
                 **kwargs):
        super().__init__()
        self._run_name = name
        self.id = id
        self._save_dir = save_dir
        self.project = project
        self.entity = entity
        self.plot_images = plot_images
        self.save_images = save_images
        self.save_HDR = save_HDR
        self.log_folder = log_folder

        self.module_logger = init_logger()

    def get_checkpoint_path(self):
        ckpt_dir = os.path.join(self.save_dir, self.project, self.id or "", "checkpoints")

        if not os.path.exists(ckpt_dir):
            return None

        checkpoints = list(reversed(sorted(os.listdir(ckpt_dir))))
        latest_checkpoint = checkpoints[0]
        self.module_logger.info(
            f"{len(checkpoints)} checkpoints found ({checkpoints}), using the latest one: {latest_checkpoint}!")
        return os.path.join(ckpt_dir, latest_checkpoint)

    def log(self, data_dict, **kwargs):
        for key, data in data_dict.items():
            if isinstance(data, wandb.Histogram):
                self.module_logger.info(f"{key}: {data.histogram}")
            elif isinstance(data, wandb.Image):
                self.log_image(data.image, name=key)
            elif isinstance(data, list) and isinstance(data[0], wandb.Image):
                for i, img in enumerate(data):
                    self.log_image(img.image, name=f"{key}_{i}")
            elif isinstance(data, wandb.Video):
                self.log_video(data, name=key)
            elif isinstance(data, wandb.Table):
                self.module_logger.info(f"{key}: {data.data}")
            else:
                self.module_logger.info(f"{key}: {data}")

    def log_video(self, video, name=None):
        import moviepy.editor as mpy
        if self.save_images:
            if self.log_folder is not None:
                img_path = os.path.join(self.log_folder, f"{name}_.mp4")
            else:
                img_path = os.path.join(self.save_dir, "media", "images", f"{name}_.mp4")
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            self.module_logger.info(f"Saving video to {img_path}")

            tensor = video._prepare_video(video.data)
            clip = mpy.ImageSequenceClip(list(tensor), fps=video._fps)
            clip.write_videofile(img_path)

    def log_image(self, image, name=None):
        if self.plot_images:
            image.show(title=name)
            # cv2.imshow(name, np.asarray(image)[..., ::-1])
            # cv2.waitKey(0)
        if self.save_images:
            if self.log_folder is not None:
                img_path = os.path.join(self.log_folder, f"{name}_.png")
            else:
                img_path = os.path.join(self.save_dir, "media", "images", f"{name}_.png")
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            self.module_logger.info(f"Saving image to {img_path}")
            image.save(img_path)

    def log_hdr(self, image_tensor, name=None):
        if self.save_HDR:
            if self.log_folder is not None:
                img_path = os.path.join(self.log_folder, f"{name}_.exr")
            else:
                img_path = os.path.join(self.save_dir, "media", "images", f"{name}_.exr")
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            self.module_logger.info(f"Saving HDR image to {img_path}")
            save_exr_image(image_tensor.cpu().permute(1, 2, 0).numpy(), img_path)

    @property
    def name(self):
        return "ConsoleLogger"

    @property
    def save_dir(self):
        """Return the root directory where experiment logs get saved, or `None` if the logger does not save data
        locally."""
        return self._save_dir

    @property
    def version(self):
        # Return the experiment version, int or str.
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        self.log(metrics)

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        pass

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        pass
