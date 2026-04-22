
import einops
import mitsuba
import numpy as np
from omegaconf import OmegaConf
import torch_scatter
import trimesh

from iif.component.model.emitter import SLFEmitter
from iif.utils.config import range2list

mitsuba.set_variant('cuda_ad_rgb')

import math
import hydra
import glob
import os
import kornia
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torchvision
from diffusers import DDIMScheduler
from iif.component.task.single_view_prediction.pipeline_rgb2x import StableDiffusionAOVMatEstPipeline
from iif.task.task import Task
from iif.utils.image_io import load_ldr_image, save_image, show_image
from iif.utils.logging import init_logger
from iif.component.model.slf import VoxelSLF
from iif.component.rendering.path_tracing import path_tracing_det_diff, path_tracing_det_spec, ray_intersect


class ChacheShading(Task):
    TASK_NAME = "3_cache_shading/iris"

    def __init__(self,
                 input,
                 output,
                 model_cfg,
                 **kwargs):
        self.module_logger = init_logger()
        super().__init__()
        
        self.input = input
        self.output = output
        self.model_cfg = model_cfg

        # Load the mesh
        mesh_path = self.input["scene_path"]
        assert os.path.exists(mesh_path), f"Mesh not found: {mesh_path}"
        self.scene = mitsuba.load_dict({
            'type': 'scene',
            'shape_id':{
                'type': os.path.splitext(mesh_path)[-1].replace('.', ''),
                'filename': mesh_path
            }
        })

        # Load the dataset
        self.dataset = hydra.utils.instantiate(self.input["dataset_cfg"])

        # Load the SLF model
        self.slf = hydra.utils.instantiate(self.model_cfg["slf"])

        # Load the Lighting model
        self.emitter = hydra.utils.instantiate(self.model_cfg["emitter"])

        # Load the BRDF
        self.brdf = hydra.utils.instantiate(self.model_cfg["brdf"])

    def log_config(self, cfg):
        super().log_config(cfg)

    def run(self):
        # Define the device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.slf = self.slf.to(device)
        self.emitter = self.emitter.to(device)
        self.brdf = self.brdf.to(device)

        # Bake the diffuse shading
        self.bake_diffuse_shading(baking_cfg=self.model_cfg["diffuse_baking"], device=device)

        # Bake the specular shading
        self.bake_specular_shading(baking_cfg=self.model_cfg["specular_baking"], device=device)

    @torch.no_grad()
    def bake_diffuse_shading(self, baking_cfg, device):
        # Prepare output folder
        self.module_logger.info("Baking diffuse shading")
        output_path = os.path.join(self.output["folder_path"],'diffuse')
        os.makedirs(output_path,exist_ok=True)
        
        # Prepare variables
        spp = baking_cfg["spp"]
        indir_depth = baking_cfg["indir_depth"]
        batch_size = baking_cfg["batch_size"]
        img_hw = self.dataset.resolution

        # Prepare denoiser
        denoiser = mitsuba.OptixDenoiser(img_hw[::-1])
        
        # Iterate over all views and bake the shading
        im_id = 0
        for batch in tqdm(self.dataset):
            # Reshape to ray-based
            path = batch["path"]
            _, H, W = batch['rays'].shape
            batch = batch.map(lambda x: einops.rearrange(x, '... c h w -> (h w) ... c'))

            # Reshape to ray-based
            rays = batch['rays']
            rays_x,rays_d = rays[...,:3].to(device),rays[...,3:6].to(device)
            positions,normals,uvs,triangle_idxs,valid = ray_intersect(self.scene,rays_x,rays_d)
            wi = rays_d
            B = len(positions)
            L = torch.zeros(B,3,device=device)
            for b in range(math.ceil(B*1.0/batch_size)):
                b0 = b*batch_size
                b1 = min(b0+batch_size,B)
                L[b0:b1] = path_tracing_det_diff(self.scene, self.emitter, self.brdf,
                                                positions[b0:b1],wi[b0:b1],normals[b0:b1],
                                                uvs[b0:b1],triangle_idxs[b0:b1],
                                                spp, indir_depth)
            assert L.isnan().any() == False
            L = denoiser(mitsuba.TensorXf(L.reshape(*img_hw,3))).numpy()

            save_image(L, os.path.join(output_path, path.replace(".JPG", ".exr").replace(".png", ".exr")))
            im_id += 1

    @torch.no_grad()
    def bake_specular_shading(self, baking_cfg, device):
        # Prepare output folder
        self.module_logger.info("Baking specular shading")
        output_path = os.path.join(self.output["folder_path"],'specular')
        os.makedirs(output_path,exist_ok=True)
        
        # Prepare variables
        spps = baking_cfg["spp"]
        indir_depth = baking_cfg["indir_depth"]
        batch_size = baking_cfg["batch_size"]
        roughness_level = torch.linspace(0.02, 1.0, len(spps))
        img_hw = self.dataset.resolution

        # Prepare denoiser
        denoiser = mitsuba.OptixDenoiser(img_hw[::-1])
        
        # Iterate over all views and bake the shading
        im_id = 0
        for batch in tqdm(self.dataset):
            # Reshape to ray-based
            path = batch["path"]
            _, H, W = batch['rays'].shape
            batch = batch.map(lambda x: einops.rearrange(x, '... c h w -> (h w) ... c'))

            rays = batch['rays']
            rays_x,rays_d = rays[...,:3].to(device),rays[...,3:6].to(device)
            positions,normals,uvs,triangle_idxs,valid = ray_intersect(self.scene,rays_x,rays_d)
            wi = rays_d
            B = len(positions)
            L0 = torch.zeros(B,3,device=device)
            L1 = L0.clone()

            for r_idx,roughness in enumerate(roughness_level):
                # BxSx3
                spp = spps[r_idx]
                B = len(positions)
                L0 = torch.zeros(B,3,device=device)
                L1 = L0.clone()

                for b in range(math.ceil(B*1.0/batch_size)):
                    b0 = b*batch_size
                    b1 = min(b0+batch_size,B)
                    L0_,L1_ = path_tracing_det_spec(self.scene, self.emitter, self.brdf,
                                                    roughness,
                                                    positions[b0:b1],wi[b0:b1],normals[b0:b1],
                                                    uvs[b0:b1],triangle_idxs[b0:b1],
                                                    spp, indir_depth)
                    L0[b0:b1] = L0_
                    L1[b0:b1] = L1_
                assert L0.isnan().any() == False
                assert L1.isnan().any() == False
                L0 = denoiser(mitsuba.TensorXf(L0.reshape(*img_hw,3))).numpy()
                L1 = denoiser(mitsuba.TensorXf(L1.reshape(*img_hw,3))).numpy()

                save_image(L0, os.path.join(output_path, f"{path.replace('.JPG', '').replace('.png', '')}_0_{r_idx}.exr"))
                save_image(L1, os.path.join(output_path, f"{path.replace('.JPG', '').replace('.png', '')}_1_{r_idx}.exr"))
            im_id += 1
