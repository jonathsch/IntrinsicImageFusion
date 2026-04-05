
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
from iif.component.rendering.path_tracing import ray_intersect


class InitEmitter(Task):
    TASK_NAME = "3_get_emitter/iris"

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

        # Get Mesh vertices and triangles
        mesh = trimesh.load_mesh(mesh_path)
        self.vertices = torch.from_numpy(np.array(mesh.vertices)).float() #(v, 3)
        self.faces = torch.from_numpy(np.array(mesh.faces)) #(f, 3)

        # Load the SLF model
        self.slf = hydra.utils.instantiate(self.model_cfg["slf"])

        # Load the CRF model
        self.crf = hydra.utils.instantiate(self.model_cfg["crf"])

    def log_config(self, cfg):
        # Implement logging logic here
        out_folder = self.output["folder_path"]
        os.makedirs(out_folder, exist_ok=True)

        # Save Emitter config
        self.model_cfg["emitter"]["cfg"]["n_triangles"] = len(self.faces)
        self.model_cfg["emitter"]["cfg"]["grid_size"] = self.slf.grid_size

        OmegaConf.save(self.model_cfg["emitter"]["cfg"], os.path.join(out_folder, f"emitter.yaml"))

    def run(self):
        # Define the device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Collect triangle radiances
        n_face = len(self.faces)
        # triangle_radiance = torch.ones(n_face, 3)
        triangle_radiance = torch.zeros(n_face, 3)
        triangle_count = torch.zeros(n_face)
        for batch in tqdm(self.dataset):
            rays = batch['rays']
            rays_x,rays_d = rays[...,:3].to(device),rays[...,3:6].to(device)

            # # Sample within pixel
            # rays_x_subpixel = rays_x.unsqueeze(1).repeat(1, SPP, 1).view(-1, xs_pixel.shape[1])
            # ds_subpixel = ds_pixel.unsqueeze(1).repeat(1, SPP, 1).view(-1, ds_pixel.shape[1])
            # dxdu_subpixel = dxdu_pixel.unsqueeze(1).repeat(1, SPP, 1).view(-1, dxdu_pixel.shape[1])
            # dydv_subpixel = dydv_pixel.unsqueeze(1).repeat(1, SPP, 1).view(-1, dydv_pixel.shape[1])

            # du,dv = torch.rand(2, len(xs_subpixel), 1, device=xs_subpixel.device) - 0.5
            # ds_subpixel = F.normalize(ds_subpixel + dxdu_subpixel * du + dydv_subpixel * dv, dim=1)

            positions,normals,uvs,triangle_idxs,valid = ray_intersect(self.scene, rays_x, rays_d)

            triangle_idxs = triangle_idxs[valid].cpu()
            radiance = batch['rgbs_ldr'][valid.cpu()]

            exposure = batch['exposure'][valid.cpu()]
            radiance = self.crf.inverse(radiance, exposure)
            # segmentation = batch['segmentation'][valid.cpu()]
            # seg_idxs, inv_idxs = segmentation.unique(return_inverse=True)

        #     triangle_radiance = torch_scatter.scatter(
        #         radiance, triangle_idxs, 0, triangle_radiance, reduce='min'
        #     )
        #     triangle_count = torch_scatter.scatter(
        #         torch.ones(len(triangle_idxs)), triangle_idxs, 0, triangle_count, reduce='sum'
        #     )

        # triangle_radiance_aggregate = triangle_radiance
        # triangle_radiance_aggregate[triangle_count == 0] = 0.
        # self.module_logger.info(f"Triangle radiance aggregate: {triangle_radiance_aggregate}")
        # triangle_radiance_aggregate_colored = triangle_radiance_aggregate.clone().mean(dim=-1, keepdim=True)
        # triangle_radiance_aggregate = torch.max(triangle_radiance_aggregate, dim=-1)[0]  # max value among 3 channels

            triangle_radiance = torch_scatter.scatter(
                radiance, triangle_idxs, 0, triangle_radiance, reduce='sum'
            )
            triangle_count = torch_scatter.scatter(
                torch.ones(len(triangle_idxs)), triangle_idxs, 0, triangle_count, reduce='sum'
            )

        triangle_radiance_aggregate = triangle_radiance / triangle_count.unsqueeze(-1).clamp_min(1) #(f, 3)
        triangle_radiance_aggregate_colored = triangle_radiance_aggregate.clone()
        triangle_radiance_aggregate = torch.max(triangle_radiance_aggregate, dim=-1)[0] # max value among 3 channels

        # Segment the emitters
        is_emitter = triangle_radiance_aggregate > self.model_cfg["threshold"]
        n_emitters = is_emitter.sum().item()
        self.module_logger.info(f"Found {n_emitters} emitter triangles out of {n_face} triangles.")
        emitter_vertices = self.vertices[self.faces[is_emitter]]
        emitter_area = torch.cross(emitter_vertices[:,1]-emitter_vertices[:,0],
                                   emitter_vertices[:,2]-emitter_vertices[:,0],-1)
        emitter_normal = F.normalize(emitter_area,dim=-1)
        emitter_area = emitter_area.norm(dim=-1)/2.0
        emitter_radiance = torch.zeros(n_emitters, 3) + 1e-2  # Init with minimal emission to avoid vanishing gradients
        # emitter_radiance = torch.zeros(n_emitters, 3)
        # Create emitter model
        self.model_cfg["emitter"]["cfg"]
        emitter = hydra.utils.instantiate(self.model_cfg["emitter"]["cfg"])

        emitter.initialize(is_emitter=is_emitter,
                           emitter_vertices=emitter_vertices,
                           emitter_area=emitter_area,
                           emitter_normal=emitter_normal,
                           emitter_radiance=emitter_radiance,
                           slf=self.slf)
        emitter = emitter.to(device)
        
        # Visualize the baked slf
        dataset_vis_cfg = self.input["dataset_cfg"]
        dataset_vis_cfg.update(self.output["visualization"]["dataset_overrides"])
        dataset_vis = hydra.utils.instantiate(dataset_vis_cfg)
        self.visualize_emitter(emitter, triangle_radiance_aggregate_colored, self.scene, dataset_vis, device)

        # Save the Emitter Segmentation
        out_folder = self.output["folder_path"]
        os.makedirs(out_folder, exist_ok=True)

        out_path = os.path.join(out_folder, f"emitter.pt")
        torch.save(emitter.state_dict() ,out_path)
        self.module_logger.info(f"Saved Emitter model to {out_path}")


    @torch.no_grad()
    def visualize_emitter(self, emitter, triangle_radiance_aggregate, scene, dataset, device):
        self.module_logger.info("Visualizing emitters")
        out_folder = os.path.join(self.output["folder_path"], self.output["visualization"]["out_folder"])
        os.makedirs(out_folder, exist_ok=True)

        for idx in tqdm(range2list(self.output["visualization"]["indices"])):
            batch = dataset[idx].to(device)
            # Reshape to ray-based
            if batch['rays'].ndim > 2:
                # Assume image-based batch
                ray_based_batch = False
                _, H, W = batch['rays'].shape
                batch = batch.map(lambda x: einops.rearrange(x, '... c h w -> (h w) ... c'))
            else:
                # Assume ray-based batch
                ray_based_batch = True
                batch = batch.map(lambda x: einops.rearrange(x, 'r ... c -> r ... c'))

            rays = batch['rays']
            radiance = batch['rgbs_ldr']
            exposure = batch['exposure']
            xs = rays[...,:3].to(device)
            ds = rays[...,3:6].to(device)

            positions,_,_,triangle_idx,valid = ray_intersect(scene,xs,ds)
            if not valid.any():
                continue

            slf = emitter(positions)
            slf = self.crf(slf.cpu(), exposure.cpu())
            emission, _, _, emission_mask = emitter.eval_emitter(positions, ds, triangle_idx)

            # Reshape to image-based
            radiance = einops.rearrange(radiance, '(h w) c -> c h w', h=H, w=W)
            emission = einops.rearrange(emission, '(h w) c -> c h w', h=H, w=W)
            emission_mask = einops.rearrange(emission_mask.float(), '(h w) -> 1 h w', h=H, w=W)
            slf = einops.rearrange(slf, '(h w) c -> c h w', h=H, w=W)

            # Save the visualization
            save_image(radiance.clamp(0,1), os.path.join(out_folder, f"{idx:04d}_gt.png"))
            save_image(emission.clamp(0,1), os.path.join(out_folder, f"{idx:04d}_emission.png"))
            save_image(emission_mask.clamp(0,1), os.path.join(out_folder, f"{idx:04d}_emission_mask.png"))
            save_image(slf.clamp(0,1), os.path.join(out_folder, f"{idx:04d}_slf.png"))

            # Visualize aggregated triangle radiance
            aggregated_radiance = triangle_radiance_aggregate[triangle_idx.cpu()]
            aggregated_radiance = einops.rearrange(aggregated_radiance, '(h w) c -> c h w', h=H, w=W)
            save_image(aggregated_radiance.clamp(0,1), os.path.join(out_folder, f"{idx:04d}_aggregated_radiance.png"))


class InitEmitterMaskCluster(Task):
    TASK_NAME = "3_get_emitter/iris"

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

        # Get Mesh vertices and triangles
        mesh = trimesh.load_mesh(mesh_path)
        self.vertices = torch.from_numpy(np.array(mesh.vertices)).float() #(v, 3)
        self.faces = torch.from_numpy(np.array(mesh.faces)) #(f, 3)

        # Load the SLF model
        self.slf = hydra.utils.instantiate(self.model_cfg["slf"])

        # Load the CRF model
        self.crf = hydra.utils.instantiate(self.model_cfg["crf"])

    def log_config(self, cfg):
        # Implement logging logic here
        out_folder = self.output["folder_path"]
        os.makedirs(out_folder, exist_ok=True)

        # Save Emitter config
        self.model_cfg["emitter"]["cfg"]["n_triangles"] = len(self.faces)
        self.model_cfg["emitter"]["cfg"]["grid_size"] = self.slf.grid_size

        OmegaConf.save(self.model_cfg["emitter"]["cfg"], os.path.join(out_folder, f"emitter.yaml"))

    def run(self):
        # Define the device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create emitter model
        self.model_cfg["emitter"]["cfg"]
        emitter = hydra.utils.instantiate(self.model_cfg["emitter"]["cfg"])

        # Collect triangle radiances
        n_face = len(self.faces)
        # triangle_radiance = torch.ones(n_face, 3)
        triangle_radiance = torch.zeros(n_face, 3)
        triangle_count = torch.zeros(n_face)

        envmap_radiance = torch.zeros_like(emitter.envmap_radiance)
        envmap_count = torch.zeros_like(emitter.envmap_radiance[...,0])
        for batch in tqdm(self.dataset):
            rays = batch['rays']
            rays_x,rays_d = rays[...,:3].to(device),rays[...,3:6].to(device)

            # # Sample within pixel
            # rays_x_subpixel = rays_x.unsqueeze(1).repeat(1, SPP, 1).view(-1, xs_pixel.shape[1])
            # ds_subpixel = ds_pixel.unsqueeze(1).repeat(1, SPP, 1).view(-1, ds_pixel.shape[1])
            # dxdu_subpixel = dxdu_pixel.unsqueeze(1).repeat(1, SPP, 1).view(-1, dxdu_pixel.shape[1])
            # dydv_subpixel = dydv_pixel.unsqueeze(1).repeat(1, SPP, 1).view(-1, dydv_pixel.shape[1])

            # du,dv = torch.rand(2, len(xs_subpixel), 1, device=xs_subpixel.device) - 0.5
            # ds_subpixel = F.normalize(ds_subpixel + dxdu_subpixel * du + dydv_subpixel * dv, dim=1)

            positions,normals,uvs,triangle_idxs,valid = ray_intersect(self.scene, rays_x, rays_d)

            triangle_idxs = triangle_idxs[valid].cpu()
            radiance = batch['rgbs_ldr'][valid.cpu()]

            exposure = batch['exposure'][valid.cpu()]
            radiance = self.crf.inverse(radiance, exposure)

            envmap_directions = rays_d[~valid].cpu()
            envmap_rad = batch['rgbs_ldr'][~valid.cpu()]
            envmap_exposure = batch['exposure'][~valid.cpu()]
            envmap_rad = self.crf.inverse(envmap_rad, envmap_exposure)
            if emitter.n_channels == 1:
                envmap_rad = envmap_rad.mean(dim=-1, keepdim=True)
            envmap_idx = emitter.envmap_dir_to_idx(envmap_directions)
            # segmentation = batch['segmentation'][valid.cpu()]
            # seg_idxs, inv_idxs = segmentation.unique(return_inverse=True)

        #     triangle_radiance = torch_scatter.scatter(
        #         radiance, triangle_idxs, 0, triangle_radiance, reduce='min'
        #     )
        #     triangle_count = torch_scatter.scatter(
        #         torch.ones(len(triangle_idxs)), triangle_idxs, 0, triangle_count, reduce='sum'
        #     )

        # triangle_radiance_aggregate = triangle_radiance
        # triangle_radiance_aggregate[triangle_count == 0] = 0.
        # self.module_logger.info(f"Triangle radiance aggregate: {triangle_radiance_aggregate}")
        # triangle_radiance_aggregate_colored = triangle_radiance_aggregate.clone().mean(dim=-1, keepdim=True)
        # triangle_radiance_aggregate = torch.max(triangle_radiance_aggregate, dim=-1)[0]  # max value among 3 channels

            triangle_radiance = torch_scatter.scatter(
                radiance, triangle_idxs, 0, triangle_radiance, reduce='sum'
            )
            triangle_count = torch_scatter.scatter(
                torch.ones(len(triangle_idxs)), triangle_idxs, 0, triangle_count, reduce='sum'
            )

            envmap_radiance = torch_scatter.scatter(
                envmap_rad, envmap_idx, 0, envmap_radiance, reduce='sum'
            )
            envmap_count = torch_scatter.scatter(
                torch.ones(len(envmap_idx)), envmap_idx, 0, envmap_count, reduce='sum'
            )



        triangle_radiance_aggregate = triangle_radiance / triangle_count.unsqueeze(-1).clamp_min(1) #(f, 3)
        triangle_radiance_aggregate_colored = triangle_radiance_aggregate.clone()
        triangle_radiance_aggregate = torch.max(triangle_radiance_aggregate, dim=-1)[0] # max value among 3 channels

        # Segment the emitters
        is_emitter = triangle_radiance_aggregate > self.model_cfg["threshold"]
        n_emitters = is_emitter.sum().item()
        self.module_logger.info(f"Found {n_emitters} emitter triangles out of {n_face} triangles.")
        emitter_vertices = self.vertices[self.faces[is_emitter]]
        emitter_area = torch.cross(emitter_vertices[:,1]-emitter_vertices[:,0],
                                   emitter_vertices[:,2]-emitter_vertices[:,0],-1)
        emitter_normal = F.normalize(emitter_area,dim=-1)
        emitter_area = emitter_area.norm(dim=-1)/2.0
        emitter_radiance = torch.zeros(n_emitters, 3) + 1e-2  # Init with minimal emission to avoid vanishing gradients
        # emitter_radiance = torch.zeros(n_emitters, 3)

        # Segment the envmap
        envmap_radiance_aggregate = envmap_radiance / envmap_count.clamp_min(1).unsqueeze(-1)
        envmap_radiance_aggregate = torch.max(envmap_radiance_aggregate, dim=-1)[0] # max value among all channels

        is_envmap_emitter = envmap_radiance_aggregate > self.model_cfg["envmap_threshold"]
        envmap_radiance[is_envmap_emitter] = 1e-2 
        envmap_radiance[~is_envmap_emitter] = 0.0
        valid_envmap_pixel = is_envmap_emitter


        emitter.initialize(is_emitter=is_emitter,
                           emitter_vertices=emitter_vertices,
                           emitter_area=emitter_area,
                           emitter_normal=emitter_normal,
                           emitter_radiance=emitter_radiance,
                           slf=self.slf,
                           envmap_radiance=envmap_radiance,
                           envmap_resolution=emitter.envmap_resolution,
                           valid_envmap_pixel=valid_envmap_pixel)
        emitter = emitter.to(device)
        
        # Visualize the baked slf
        dataset_vis_cfg = self.input["dataset_cfg"]
        dataset_vis_cfg.update(self.output["visualization"]["dataset_overrides"])
        dataset_vis = hydra.utils.instantiate(dataset_vis_cfg)
        self.visualize_emitter(emitter, triangle_radiance_aggregate_colored, self.scene, dataset_vis, device)

        # Save the Emitter Segmentation
        out_folder = self.output["folder_path"]
        os.makedirs(out_folder, exist_ok=True)

        out_path = os.path.join(out_folder, f"emitter.pt")
        torch.save(emitter.state_dict() ,out_path)
        self.module_logger.info(f"Saved Emitter model to {out_path}")


    @torch.no_grad()
    def visualize_emitter(self, emitter, triangle_radiance_aggregate, scene, dataset, device):
        self.module_logger.info("Visualizing emitters")
        out_folder = os.path.join(self.output["folder_path"], self.output["visualization"]["out_folder"])
        os.makedirs(out_folder, exist_ok=True)

        for idx in tqdm(range2list(self.output["visualization"]["indices"])):
            batch = dataset[idx].to(device)
            # Reshape to ray-based
            if batch['rays'].ndim > 2:
                # Assume image-based batch
                ray_based_batch = False
                _, H, W = batch['rays'].shape
                batch = batch.map(lambda x: einops.rearrange(x, '... c h w -> (h w) ... c'))
            else:
                # Assume ray-based batch
                ray_based_batch = True
                batch = batch.map(lambda x: einops.rearrange(x, 'r ... c -> r ... c'))

            rays = batch['rays']
            radiance = batch['rgbs_ldr']
            exposure = batch['exposure']
            xs = rays[...,:3].to(device)
            ds = rays[...,3:6].to(device)

            positions,_,_,triangle_idx,valid = ray_intersect(scene,xs,ds)
            if not valid.any():
                continue

            slf = emitter(positions)
            slf = self.crf(slf.cpu(), exposure.cpu())
            emission, _, _, emission_mask = emitter.eval_emitter(positions, ds, triangle_idx)

            # Reshape to image-based
            radiance = einops.rearrange(radiance, '(h w) c -> c h w', h=H, w=W)
            emission = einops.rearrange(emission, '(h w) c -> c h w', h=H, w=W)
            emission_mask = einops.rearrange(emission_mask.float(), '(h w) -> 1 h w', h=H, w=W)
            slf = einops.rearrange(slf, '(h w) c -> c h w', h=H, w=W)

            # Save the visualization
            save_image(radiance.clamp(0,1), os.path.join(out_folder, f"{idx:04d}_gt.png"))
            save_image(emission.clamp(0,1), os.path.join(out_folder, f"{idx:04d}_emission.png"))
            save_image(emission_mask.clamp(0,1), os.path.join(out_folder, f"{idx:04d}_emission_mask.png"))
            save_image(slf.clamp(0,1), os.path.join(out_folder, f"{idx:04d}_slf.png"))

            # Visualize aggregated triangle radiance
            aggregated_radiance = triangle_radiance_aggregate[triangle_idx.cpu()]
            aggregated_radiance = einops.rearrange(aggregated_radiance, '(h w) c -> c h w', h=H, w=W)
            save_image(aggregated_radiance.clamp(0,1), os.path.join(out_folder, f"{idx:04d}_aggregated_radiance.png"))

            # Visualize the envmap
            envmap_radiance = emitter.envmap_radiance.cpu()
            envmap_radiance = envmap_radiance.reshape(emitter.envmap_resolution[0], emitter.envmap_resolution[1], emitter.n_channels)
            save_image(envmap_radiance.permute(2,0,1).clamp(0,1), os.path.join(out_folder, f"{idx:04d}_envmap.png"))
