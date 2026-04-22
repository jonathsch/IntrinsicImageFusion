
import einops
import mitsuba
import numpy as np
from omegaconf import OmegaConf
import trimesh

from iif.utils.config import range2list
from iif.utils.model import freeze_model
mitsuba.set_variant('cuda_ad_rgb')

import math
import hydra
import glob
import os
import kornia
import torch
from tqdm import tqdm
import torchvision
from diffusers import DDIMScheduler
from iif.component.task.single_view_prediction.pipeline_rgb2x import StableDiffusionAOVMatEstPipeline
from iif.task.task import Task
from iif.utils.image_io import load_ldr_image, save_image, show_image
from iif.utils.logging import init_logger
from iif.component.model.slf import VoxelSLF
from iif.component.rendering.path_tracing import ray_intersect, ray_intersect_w_depth


class BakingSLF(Task):
    TASK_NAME = "0_baking_slf/iris"

    def __init__(self,
                 input,
                 output,
                 model_cfg,
                 **kwargs):
        super().__init__()
        
        self.input = input
        self.output = output
        self.model_cfg = model_cfg

        self.module_logger = init_logger()

    def log_config(self, cfg):
        out_folder = self.output["folder_path"]
        os.makedirs(out_folder, exist_ok=True)

        # Save CRF config
        if isinstance(self.model_cfg["crf"]["cfg"], str):
            self.model_cfg["crf"]["cfg"] = OmegaConf.load(self.model_cfg["crf"]["cfg"])
        OmegaConf.save(self.model_cfg["crf"]["cfg"], os.path.join(out_folder, f"crf.yaml"))

        # Save SLF config
        if isinstance(self.model_cfg["slf"]["cfg"], str):
            self.model_cfg["slf"]["cfg"] = OmegaConf.load(self.model_cfg["slf"]["cfg"])
        OmegaConf.save(self.model_cfg["slf"]["cfg"], os.path.join(out_folder, f"slf.yaml"))

    def run(self):
        # Define the device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the mesh
        mesh_path = self.input["scene_path"]
        assert os.path.exists(mesh_path), f"Mesh not found: {mesh_path}"
        scene = mitsuba.load_dict({
            'type': 'scene',
            'shape_id':{
                'type': os.path.splitext(mesh_path)[-1].replace('.', ''),
                'filename': mesh_path
            }
        })
        mesh = trimesh.load_mesh(mesh_path)

        # Load the dataset
        dataset = hydra.utils.instantiate(self.input["dataset_cfg"])

        # Create the CRF model
        crf = hydra.utils.instantiate(self.model_cfg["crf"]["cfg"])
        crf = crf.to(device)
        freeze_model(crf)

        # Create the SLF model
        slf = hydra.utils.instantiate(self.model_cfg["slf"]["cfg"])

        if self.model_cfg["slf"]["pt"] is not None:
            slf.load_state_dict(torch.load(self.model_cfg["slf"]["pt"], weights_only=True))
        else:
            self.module_logger.info("No pre-trained SLF model provided, initializing from scratch.")
             # Find Scene BBox
            # voxel_min, voxel_max = self.get_scene_bbox(mesh)
            voxel_min, voxel_max = self.get_scene_bbox(scene, dataset, device)

            # Find occupied voxels
            mask = self.find_occupied_voxels(scene, dataset, device, voxel_min, voxel_max, resolution=slf.grid_size)

            # Inintialize surface light field
            slf.initialize(mask.cpu(), voxel_min.item(), voxel_max.item())
        slf = slf.to(device)
        freeze_model(slf)

        # Create surface light field
        slf = self.bake_slf(slf, scene, dataset, device, crf)

        # Visualize the baked slf
        dataset_vis_cfg = self.input["dataset_cfg"]
        dataset_vis_cfg.update(self.output["visualization"]["dataset_overrides"])
        dataset_vis = hydra.utils.instantiate(dataset_vis_cfg)
        self.visualize_slf(slf, scene, dataset_vis, device, crf)

        # Save the SLF and CRF
        out_folder = self.output["folder_path"]
        os.makedirs(out_folder, exist_ok=True)
        
        torch.save(slf.state_dict(), os.path.join(out_folder, f"slf.pt"))
        self.module_logger.info(f"Saved SLF to {out_folder}")

        torch.save(crf.state_dict(), os.path.join(out_folder, f"crf.pt"))
        self.module_logger.info(f"Saved CRF to {out_folder}")

    @torch.no_grad()
    def get_scene_bbox(self, scene, dataset, device):
        self.module_logger.info("Finding scene bounding box")
        voxel_min = math.inf
        voxel_max = -math.inf
        for idx in tqdm(range(len(dataset))):
            batch = dataset[idx]

            # Reshape to ray-based
            if batch['rays'].ndim > 2:
                # Assume image-based batch
                ray_based_batch = False
                _, H, W = batch['rays'].shape
                batch = batch.map(lambda x: einops.rearrange(x, '... c h w -> (h w) ... c'))    

            rays = batch['rays']
            xs = rays[...,:3].to(device)
            ds = rays[...,3:6].to(device)

            positions,_,_,_,valid = ray_intersect(scene,xs,ds)
            if not valid.any():
                continue

            position = positions[valid]
            voxel_min = min(voxel_min,position.min())
            voxel_max = max(voxel_max,position.max())

        extent = voxel_max - voxel_min

        voxel_min = voxel_min - 0.01 * extent
        voxel_max = voxel_max + 0.01 * extent

        self.module_logger.info(f"Scene bbox min: {voxel_min}, max: {voxel_max}")

        assert voxel_max > voxel_min, "Invalid scene bounding box"
        assert voxel_min != math.inf and voxel_max != -math.inf, "No valid intersections found in the dataset"
        return voxel_min, voxel_max

    # @torch.no_grad()
    # def get_scene_bbox(self, mesh):
    #     self.module_logger.info("Finding scene bounding box")
    #     pos_min = mesh.vertices.min()
    #     pos_max = mesh.vertices.max()

    #     extent = pos_max - pos_min
    #     voxel_min = pos_min - 0.01 * extent
    #     voxel_max = pos_max + 0.01 * extent

    #     self.module_logger.info(f"Scene bbox min: {voxel_min}, max: {voxel_max}")

    #     assert voxel_max >= voxel_min, "Invalid scene bounding box"
    #     return voxel_min, voxel_max
    
    @torch.no_grad()
    def find_occupied_voxels(self, scene, dataset, device, voxel_min, voxel_max, resolution):
        self.module_logger.info("Finding occupied voxels")
        spatial_hist = torch.zeros(resolution**3,device=device)
        for idx in tqdm(range(len(dataset))):
            batch = dataset[idx]

            # Reshape to ray-based
            if batch['rays'].ndim > 2:
                # Assume image-based batch
                ray_based_batch = False
                _, H, W = batch['rays'].shape
                batch = batch.map(lambda x: einops.rearrange(x, '... c h w -> (h w) ... c'))

            rays = batch['rays']
            xs = rays[...,:3].to(device)
            ds = rays[...,3:6].to(device)

            positions,_,_,_,valid = ray_intersect(scene,xs,ds)
            if not valid.any():
                continue
            
            position = (positions[valid]-voxel_min)/(voxel_max-voxel_min)
            position = (position*resolution).long().clamp(0,resolution-1)
            inds = position[...,0] + position[...,1]*resolution\
                + position[...,2]*resolution*resolution
            spatial_hist.scatter_add_(0,inds,torch.ones_like(inds).float())
        spatial_hist = spatial_hist.reshape(resolution,resolution,resolution)

        return spatial_hist > 0
    
    @torch.no_grad()
    def bake_slf(self, slf, scene, dataset, device, crf):
        self.module_logger.info("Baking surface light field")
        for idx in tqdm(range(len(dataset))):
            batch = dataset[idx].to(device)

            # Reshape to ray-based
            if batch['rays'].ndim > 2:
                # Assume image-based batch
                ray_based_batch = False
                _, H, W = batch['rays'].shape
                batch = batch.map(lambda x: einops.rearrange(x, '... c h w -> (h w) ... c'))

            rays = batch['rays']
            radiance = batch['rgbs_ldr']
            xs = rays[...,:3].to(device)
            ds = rays[...,3:6].to(device)

            exposure = batch['exposure']
            radiance = crf.inverse(radiance, exposure).to(device)

            positions,_,_,_,valid = ray_intersect(scene,xs,ds)
            if not valid.any():
                continue

            slf.scatter_add(positions[valid],radiance[valid])


        # average pooling the radiance
        slf.compute()

        return slf
    
    @torch.no_grad()
    def visualize_slf(self, slf, scene, dataset, device, crf):
        self.module_logger.info("Visualizing baked surface light field")
        out_folder = os.path.join(self.output["folder_path"], self.output["visualization"]["out_folder"])
        os.makedirs(out_folder, exist_ok=True)

        for idx in tqdm(range2list(self.output["visualization"]["indices"], len(dataset))):
            batch = dataset[idx]
            path = batch["path"]
            
            batch = batch.to(device)
            # Reshape to ray-based
            if batch['rays'].ndim > 2:
                # Assume image-based batch
                ray_based_batch = False
                _, H, W = batch['rays'].shape
                batch = batch.map(lambda x: einops.rearrange(x, '... c h w -> (h w) ... c'))

            rays = batch['rays']
            radiance = batch['rgbs_ldr']
            xs = rays[...,:3].to(device)
            ds = rays[...,3:6].to(device)

            exposure = batch['exposure']
            radiance = crf.inverse(radiance, exposure).to(device)

            positions,normals,_,_,valid, depth = ray_intersect_w_depth(scene,xs,ds)
            if not valid.any():
                continue

            pred_radiance = slf(positions)["rgb"]

            # Reshape to image-based
            radiance = einops.rearrange(radiance, '(h w) c -> c h w', h=H, w=W)
            pred_radiance = einops.rearrange(pred_radiance, '(h w) c -> c h w', h=H, w=W)
            normals = einops.rearrange(normals, '(h w) c -> c h w', h=H, w=W)
            depth = einops.rearrange(depth, '(h w) -> h w', h=H, w=W)

            # Save the visualization
            save_image(radiance.clamp(0,1), os.path.join(out_folder, f"{idx:04d}_gt.png"))
            save_image(pred_radiance.clamp(0,1), os.path.join(out_folder, f"{idx:04d}_slf.png"))
            save_image(normals.clamp(0,1), os.path.join(out_folder, f"{idx:04d}_normal.png"))

            save_image(normals.permute(1,2,0).numpy(force=True), os.path.join(out_folder, "normal", os.path.basename(path).replace(".png", ".exr")))
            save_image(depth.unsqueeze(0).numpy(force=True), os.path.join(out_folder, "depth", os.path.basename(path).replace(".png", ".exr")))
