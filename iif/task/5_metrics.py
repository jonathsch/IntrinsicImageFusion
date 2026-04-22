
import json
import einops
import mitsuba
import numpy as np

from iif.utils.model import get_config
mitsuba.set_variant('cuda_ad_rgb')
from omegaconf import OmegaConf

from iif.utils.datastructure import Batch
import math
import hydra
import glob
import os
import kornia
import torch
from tqdm import tqdm
import torchvision
from diffusers import DDIMScheduler
import torch.nn.functional as F
from iif.component.task.single_view_prediction.pipeline_rgb2x import StableDiffusionAOVMatEstPipeline
from iif.task.task import Task
from iif.utils.image_io import load_image, load_ldr_image, save_image, show_image
from iif.utils.logging import init_logger
from iif.component.model.slf import VoxelSLF
from iif.component.rendering.path_tracing import path_tracing, ray_intersect


class Metrics(Task):
    TASK_NAME = "metrics"
    MODALITY_TO_EXTENSION = {
        "rgbs_ldr": ".png",
        "rgbs_hdr": ".exr",
        "albedo": ".png",
        "albedo_scaled": ".png",
        "albedo_perfect_scaled": ".png",
        "roughness": ".png",
        "roughness_perfect_scaled": ".png",
        "metallic": ".png",
        "metallic_perfect_scaled": ".png",
        "emission": ".exr",
        "emission_mask": ".png",
    }

    def __init__(self,
                 input,
                 output,
                 metrics_cfg,
                 **kwargs):
        super().__init__()
        
        self.input = input
        self.output = output
        self.metrics_cfg = metrics_cfg

        self.module_logger = init_logger()

    def log_config(self, cfg):
        # Implement logging logic here
        super().log_config(cfg)

    @torch.no_grad()
    def run(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        to_tensor = torchvision.transforms.ToTensor()

        # Load the dataset
        dataset = hydra.utils.instantiate(self.input["target"]["dataset_cfg"])
        dataset_modality = self.input["target"]["modality"]

        # Get pred root folder
        pred_root = self.input["pred"]["folder_path"]
        pred_modaility = self.input["pred"]["modality"]

        # Load the metric functions
        self.metric_fns = {metric_name: hydra.utils.instantiate(metric_cfg).to(device) for metric_name, metric_cfg in self.metrics_cfg.items()}

        # =========================== Evaluate =============================

        metrics = Batch()
        # Iterate over the dataset
        for image_idx in tqdm(range(len(dataset))):
            batch = dataset[image_idx]
            file_id = batch['path'].split('/')[-1].split('.')[0]

            # Get target image
            target = batch[dataset_modality].to(device)

            # Get predicted image
            file_path = os.path.join(pred_root, pred_modaility, f"{file_id}{self.MODALITY_TO_EXTENSION[pred_modaility]}")
            pred = to_tensor(load_image(file_path)).to(device)

            # Calculate metrics
            metrics[file_id] = self.evaluate_sample(pred, target)

        # Add aggregated metrics
        metrics_aggregated = Batch.from_batch_list(*metrics.values()).map(lambda x: torch.tensor(x)).mean(dim=0).map(lambda x: x.item())

        # Save the metrics
        os.makedirs(os.path.dirname(self.output["file_path"]), exist_ok=True)
        with open(self.output["file_path"], "w") as f:
            json.dump({"aggregated": metrics_aggregated.to_dict(), "individual": metrics.to_dict()}, f, indent=4)

    @torch.no_grad()
    def evaluate_sample(self, pred, target):
        # Change shape to 3-channel
        if pred.shape[0] == 1:
            pred = pred.repeat(3, 1, 1)
        if target.shape[0] == 1:
            target = target.repeat(3, 1, 1)

        metrics = Batch()
        for metric_name, metric_fn in self.metric_fns.items():
            # Compute the metric
            metric_value = metric_fn(pred.unsqueeze(0), target.unsqueeze(0))  # Add batch dimension

            # Store the metric value
            metrics[metric_name] = metric_value.item()
        return metrics
            