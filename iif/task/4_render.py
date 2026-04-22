
import json
import einops
import mitsuba
import numpy as np

from iif.component.datamodule.dataset.fipt_synthetic import get_ray_directions, get_ray_directions_focal, get_rays
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
import torch.nn.functional as F
from moviepy import *
from iif.component.task.single_view_prediction.pipeline_rgb2x import StableDiffusionAOVMatEstPipeline
from iif.task.task import Task
from iif.utils.image_io import load_ldr_image, save_image, show_image
from iif.utils.logging import init_logger
from iif.component.model.slf import VoxelSLF
from iif.component.rendering.path_tracing import path_tracing, ray_intersect


class Render(Task):
    TASK_NAME = "render"
    MODALITY_TO_EXTENSION = {
        "rgbs_ldr": ".png",
        "rgbs_hdr": ".exr",
        "albedo": ".png",
        "albedo_nonmasked": ".png",
        "albedo_scaled": ".png",
        "albedo_perfect_scaled": ".png",
        "roughness_perfect_scaled": ".png",
        "metallic_perfect_scaled": ".png",
        "roughness": ".png",
        "metallic": ".png",
        "emission": ".exr",
        "emission_mask": ".png",
    }
    BRDF_MODALITIES = ["albedo", "albedo_scaled", "roughness", "metallic"]
    IMAGE_MODALITIES = ["rgbs_ldr", "rgbs_hdr"]

    def __init__(self,
                 input,
                 output,
                 render_cfg,
                 **kwargs):
        super().__init__()
        
        self.input = input
        self.output = output
        self.render_cfg = render_cfg

        self.module_logger = init_logger()

        self.configure_model()

    def configure_model(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ============================ Mesh =============================
        # Initialize the scene
        assert os.path.exists(self.input["scene_path"]), 'Mesh not found: '+ self.input["scene_path"]
        self.scene = mitsuba.load_dict({
            'type': 'scene',
            'shape_id':{
                'type': os.path.splitext(self.input["scene_path"])[-1].replace('.', ''),
                'filename': self.input["scene_path"]
            }
        })

        # Load the dataset
        if self.input.get("dataset_cfg", None):
            self.dataset = hydra.utils.instantiate(self.input["dataset_cfg"])
            self.im_hw = self.dataset.resolution
        else:
            self.im_hw = (self.render_cfg["resolution"]["height"], self.render_cfg["resolution"]["width"])

        # ============================ Models =============================
        # Load the BRDF
        self.brdf_cfg = get_config(self.input["brdf"])
        if self.brdf_cfg is None:
            self.brdf = None
        else:
            self.brdf = hydra.utils.instantiate(self.input["brdf"]).to(self.device)

        # Load the CRF
        self.crf_cfg = get_config(self.input["crf"])
        if self.crf_cfg is None:
            self.crf = None
        else:
            self.crf = hydra.utils.instantiate(self.input["crf"]).to(self.device)

        # Load the Emitter
        self.emitter_cfg = get_config(self.input["emitter"])
        if self.emitter_cfg is None:
            self.emitter = None
        else:
            self.emitter = hydra.utils.instantiate(self.input["emitter"]).to(self.device)

        # =========================== Render =============================
        # Set up a denoiser
        self.denoiser = mitsuba.OptixDenoiser([self.im_hw[1], self.im_hw[0]])

    def log_config(self, cfg):
        # Implement logging logic here
        super().log_config(cfg)

    @torch.no_grad()
    def run(self):
        # Seed
        torch.manual_seed(0)
        np.random.seed(0)

        # Render and eval all images
        for image_idx in tqdm(range(len(self.dataset))):
            batch = self.dataset[image_idx]
            batch = batch.to(self.device)

            # Render all the requested modalities
            rendered_outputs = self.render_single(rays=batch['rays'], 
                                                  exposure=batch['exposure'],
                                                  file_name=batch['path'].split('/')[-1].split('.')[0], 
                                                  albedo_gt=batch['albedo'] if 'albedo' in batch else None,
                                                  roughness_gt=batch['roughness'] if 'roughness' in batch else None,
                                                  metallic_gt=batch['metallic'] if 'metallic' in batch else None,
                                                  emission_gt=batch['emission'] if 'emission' in batch else None,
                                                  segmentation_gt=batch['segmentation'] if 'segmentation' in batch else None)

            # Save the renderings
            self.save_renderings(rendered_outputs)

    @torch.no_grad()
    def render_single(self, 
                      rays, 
                      exposure,
                      file_name,
                      albedo_gt=None,
                      roughness_gt=None,
                      metallic_gt=None,
                      emission_gt=None,
                      segmentation_gt=None):
        # self.module_logger.info(f"Rendering {batch}...")
        outputs = Batch()
        modalities_to_render = self.render_cfg["modalities_to_render"]

        # Get kwargs
        SPP = self.render_cfg["spp"]
        spp = self.render_cfg["spp_batch"]
        assert SPP % spp == 0, f"spp should be divisible by spp_batch, but got {SPP} and {spp}"

        # Extract data from the batch
        _, H, W = rays.shape
        rays = einops.rearrange(rays, '... c h w -> (h w) ... c')
        xs_pixel,ds_pixel = rays[:,:3], rays[:,3:6]
        ds_pixel = F.normalize(ds_pixel,dim=1)
        dxdu_pixel,dydv_pixel = rays[:,6:9],rays[:,9:12]

        # =================== Render Modalities ==================
        if self.render_cfg["jitter_within_pixel"]:
            # Find the intersection of rays with the scene
            assert not (not self.render_cfg["jitter_within_pixel"] and SPP > 1), f"Jittering within pixel is not allowed, but spp is higher than 1"
            xs_subpixel = xs_pixel.unsqueeze(1).repeat(1, spp, 1).view(-1, xs_pixel.shape[1])
            ds_subpixel = ds_pixel.unsqueeze(1).repeat(1, spp, 1).view(-1, ds_pixel.shape[1])
            dxdu_subpixel = dxdu_pixel.unsqueeze(1).repeat(1, spp, 1).view(-1, dxdu_pixel.shape[1])
            dydv_subpixel = dydv_pixel.unsqueeze(1).repeat(1, spp, 1).view(-1, dydv_pixel.shape[1])

            # Batched rendering
            lighting_modalities = None
            material_modalities = None
            for _ in range(SPP//spp):
                # Sample within pixel
                du, dv = torch.rand(2, len(xs_subpixel), 1, device=xs_subpixel.device) - 0.5
                ds_subpixel_sub = F.normalize(ds_subpixel + dxdu_subpixel * du + dydv_subpixel * dv, dim=1)

                # Ray-scene intersection
                positions_subpixel, normals_subpixel, _, triangle_idx_subpixel, valid_subpixel = ray_intersect(self.scene, xs_subpixel, ds_subpixel_sub)

                # Get modalities
                lighting_modalities_sub = self.render_lighting_modalities(positions_subpixel, ds_subpixel, triangle_idx_subpixel, valid_subpixel, spp)
                material_modalities_sub = self.render_material_modalities(positions_subpixel, triangle_idx_subpixel, valid_subpixel, spp)
                if material_modalities is None:
                    lighting_modalities = lighting_modalities_sub
                    material_modalities = material_modalities_sub
                else:
                    material_modalities = material_modalities + material_modalities_sub
                    lighting_modalities = lighting_modalities + lighting_modalities_sub
            material_modalities = material_modalities / (SPP//spp)
            lighting_modalities = lighting_modalities / (SPP//spp)
        else:
            xs_subpixel = xs_pixel
            ds_subpixel = ds_pixel

            # Ray-scene intersection
            positions_subpixel, normals_subpixel, _, triangle_idx_subpixel, valid_subpixel = ray_intersect(self.scene,xs_subpixel,ds_subpixel)

            # Get modalities
            lighting_modalities = self.render_lighting_modalities(positions_subpixel, ds_subpixel, triangle_idx_subpixel, valid_subpixel, 1)
            material_modalities = self.render_material_modalities(positions_subpixel, triangle_idx_subpixel, valid_subpixel, 1)

        # Process modalities
        if "emission" in modalities_to_render:
            outputs["emission"] = lighting_modalities["emission"]

        if "emission_mask" in modalities_to_render:
            outputs["emission_mask"] = lighting_modalities["emission_pred_mask"].unsqueeze(-1)

        if "emission_pred_mask" in lighting_modalities:
            emission_mask = lighting_modalities["emission_pred_mask"].unsqueeze(-1)
        elif "emission" in lighting_modalities:
            emission_mask = (lighting_modalities["emission"] > 0).any(dim=-1, keepdim=True).float()
        elif emission_gt is not None:
            emission_mask = (einops.rearrange(emission_gt, '... c h w -> (h w) ... c') > 0).any(dim=-1, keepdim=True).float()
        else:
            emission_mask = None

        if "albedo" in modalities_to_render:
            # Mask out the emitters in the albedo
            outputs["albedo_nonmasked"] = torch.clamp(material_modalities["albedo"], 0.0, 1.0)

            if self.render_cfg["mask_emitters"]:
                material_modalities["albedo"] *= (1 - emission_mask)

            if self.render_cfg["mask_speculars"]:
                specular_mask = material_modalities["roughness"] < 5e-2
                material_modalities["albedo"][specular_mask.squeeze(-1)] = 1.
                # material_modalities["albedo"] *= (1 - specular_mask)

            outputs["albedo"] = torch.clamp(material_modalities["albedo"], 0.0, 1.0)

        if "roughness" in modalities_to_render:
            # Mask out the emitters in the roughness
            if self.render_cfg["mask_emitters"]:
                material_modalities["roughness"] *= (1 - emission_mask)
            outputs["roughness"] = torch.clamp(material_modalities["roughness"], 0.0, 1.0)

            if self.render_cfg["perfect_scale_roughness_to_gt"]:
                segmentation_gt_perfect_scale = einops.rearrange(segmentation_gt, '... c h w -> (h w) ... c').squeeze(-1).long()
                roughness_gt_perfect_scale = einops.rearrange(roughness_gt, '... c h w -> (h w) ... c')
                roughness_pred_perfect_scale = material_modalities["roughness"].clone()

                obj_ids = segmentation_gt_perfect_scale.unique()
                obj_transforms = []
                segment_segmentation = segmentation_gt_perfect_scale.clone()
                for obj_idx, obj_id in enumerate(obj_ids):
                        mask = segmentation_gt_perfect_scale == obj_id
                        segment_segmentation[mask] = obj_idx
                        obj_transforms.append(find_affine_transform(roughness_pred_perfect_scale, roughness_gt_perfect_scale, mask))
                obj_transforms = torch.stack(obj_transforms, dim=0)  # (num_obj, 3, 4)

                obj_transforms = torch.gather(obj_transforms[None, ...].expand(segment_segmentation.shape[0], -1, -1, -1),
                                                1,
                                                segment_segmentation[:, None, None, None].expand(-1, 1, *obj_transforms.shape[1:])).squeeze(1)  # (N, 3, 4)     
                roughness_pred_perfect_scale = einops.einsum(torch.cat([roughness_pred_perfect_scale, torch.ones_like(roughness_pred_perfect_scale[..., :1])], dim=-1), obj_transforms, "B D, B C D -> B C").clamp(0,1).nan_to_num(0)

                roughness_pred_perfect_scale = torch.clamp(roughness_pred_perfect_scale, 0.0, 1.0)
                outputs["roughness_perfect_scaled"] = roughness_pred_perfect_scale

        if "metallic" in modalities_to_render:
            # Mask out the emitters in the metallic
            if self.render_cfg["mask_emitters"]:
                material_modalities["metallic"] *= (1 - emission_mask)

            outputs["metallic"] = torch.clamp(material_modalities["metallic"], 0.0, 1.0)

            if self.render_cfg["perfect_scale_metallic_to_gt"]:
                segmentation_gt_perfect_scale = einops.rearrange(segmentation_gt, '... c h w -> (h w) ... c').squeeze(-1).long()
                metallic_gt_perfect_scale = einops.rearrange(metallic_gt, '... c h w -> (h w) ... c')
                metallic_pred_perfect_scale = material_modalities["metallic"].clone()

                obj_ids = segmentation_gt_perfect_scale.unique()
                obj_transforms = []
                segment_segmentation = segmentation_gt_perfect_scale.clone()
                for obj_idx, obj_id in enumerate(obj_ids):
                        mask = segmentation_gt_perfect_scale == obj_id
                        segment_segmentation[mask] = obj_idx
                        obj_transforms.append(find_affine_transform(metallic_pred_perfect_scale, metallic_gt_perfect_scale, mask))
                obj_transforms = torch.stack(obj_transforms, dim=0)  # (num_obj, 3, 4)

                obj_transforms = torch.gather(obj_transforms[None, ...].expand(segment_segmentation.shape[0], -1, -1, -1),
                                                1,
                                                segment_segmentation[:, None, None, None].expand(-1, 1, *obj_transforms.shape[1:])).squeeze(1)  # (N, 3, 4)     
                metallic_pred_perfect_scale = einops.einsum(torch.cat([metallic_pred_perfect_scale, torch.ones_like(metallic_pred_perfect_scale[..., :1])], dim=-1), obj_transforms, "B D, B C D -> B C").clamp(0,1).nan_to_num(0)

                metallic_pred_perfect_scale = torch.clamp(metallic_pred_perfect_scale, 0.0, 1.0)
                outputs["metallic_perfect_scaled"] = metallic_pred_perfect_scale

        if "albedo" in modalities_to_render:
            if self.render_cfg["scale_albedo_to_gt"]:
                albedo_gt_scale = einops.rearrange(albedo_gt, '... c h w -> (h w) ... c')
                albedo_gt_scale_mean = albedo_gt_scale.mean(dim=0, keepdim=True)
                albedo_pixel_mean = material_modalities["albedo"].mean(dim=0, keepdim=True)
                albedo_pixel = material_modalities["albedo"] * (albedo_gt_scale_mean / (albedo_pixel_mean).clamp(min=1e-20))
                albedo_pixel = torch.clamp(albedo_pixel, 0.0, 1.0)
                outputs["albedo_scaled"] = albedo_pixel

            if self.render_cfg["perfect_scale_albedo_to_gt"]:
                segmentation_gt_perfect_scale = einops.rearrange(segmentation_gt, '... c h w -> (h w) ... c').squeeze(-1).long()
                albedo_gt_perfect_scale = einops.rearrange(albedo_gt, '... c h w -> (h w) ... c')
                albedo_pred_perfect_scale = material_modalities["albedo"].clone()

                obj_ids = segmentation_gt_perfect_scale.unique()
                obj_transforms = []
                segment_segmentation = segmentation_gt_perfect_scale.clone()
                for obj_idx, obj_id in enumerate(obj_ids):
                        mask = segmentation_gt_perfect_scale == obj_id
                        segment_segmentation[mask] = obj_idx
                        obj_transforms.append(find_affine_transform(albedo_pred_perfect_scale, albedo_gt_perfect_scale, mask))
                obj_transforms = torch.stack(obj_transforms, dim=0)  # (num_obj, 3, 4)

                obj_transforms = torch.gather(obj_transforms[None, ...].expand(segment_segmentation.shape[0], -1, -1, -1), 
                                              1, 
                                              segment_segmentation[:, None, None, None].expand(-1, 1, *obj_transforms.shape[1:])).squeeze(1)  # (N, 3, 4)
                albedo_pred_perfect_scale = einops.einsum(torch.cat([albedo_pred_perfect_scale, torch.ones_like(albedo_pred_perfect_scale[..., :1])], dim=-1), obj_transforms, "B D, B C D -> B C").clamp(0,1).nan_to_num(0)

                albedo_pred_perfect_scale = torch.clamp(albedo_pred_perfect_scale, 0.0, 1.0)
                outputs["albedo_perfect_scaled"] = albedo_pred_perfect_scale

        # =================== Render Image ==================
        if any(m in modalities_to_render for m in self.IMAGE_MODALITIES):
            # Render with Path Tracing
            L = torch.zeros_like(xs_pixel)
            for _ in range(SPP//spp):
                L += path_tracing(
                    self.scene, self.emitter, self.brdf, 
                    xs_pixel, ds_pixel, dxdu_pixel, dydv_pixel, spp,
                    depth=self.render_cfg["depth"]
                )
            L = L / (SPP//spp)
            rgbs_hdr = L 

            if self.render_cfg["denoise"]:
                rgbs_hdr = einops.rearrange(rgbs_hdr, '(h w) c -> h w c', h=H, w=W)
                rgbs_hdr = self.denoiser(rgbs_hdr.numpy(force=True)).torch().to(rgbs_hdr.device)
                rgbs_hdr = einops.rearrange(rgbs_hdr, 'h w c -> (h w) c')

            if "rgbs_hdr" in modalities_to_render:
                outputs["rgbs_hdr"] = rgbs_hdr

            if "rgbs_ldr" in modalities_to_render:
                # Apply the camera response function
                if isinstance(exposure, (int, float)):
                    exposure = torch.ones(len(rgbs_hdr), 1, device=self.device) * exposure
                else:
                    exposure = einops.rearrange(exposure, '... c h w -> (h w) ... c')
                rgbs_ldr = self.crf(rgbs_hdr, exposure)
                outputs["rgbs_ldr"] = rgbs_ldr

        # =================== Reshape Outputs ==================
        outputs = outputs.map(lambda x: einops.rearrange(x, "(h w) c -> c h w", h=H, w=W))
        outputs = Batch({
            file_name: outputs
        })
        return outputs
    
    @torch.no_grad()
    def render_lighting_modalities(self, positions_subpixel, ds_subpixel, triangle_idx_subpixel, valid_subpixel, spp):
        outputs = Batch()
        modalities_to_render = self.render_cfg["modalities_to_render"]
        
        # Render Lighting modalities
        if "emission" in modalities_to_render:
            # Query the emitter
            Le_subpixel, _, _, _ = self.emitter.eval_emitter(positions_subpixel, ds_subpixel, triangle_idx_subpixel)
            emission_pred_mask = (Le_subpixel.sum(-1) > 0).float()

            Le_subpixel = Le_subpixel * valid_subpixel.unsqueeze(-1)
            emission_pred_mask = emission_pred_mask * valid_subpixel

            Le_pixel = einops.rearrange(Le_subpixel, "(b spp) c -> b spp c", spp=spp).mean(1)
            emission_pred_mask = einops.rearrange(emission_pred_mask, "(b spp) -> b spp", spp=spp).mean(1)

            outputs["emission"] = Le_pixel
            outputs["emission_pred_mask"] = emission_pred_mask

        return outputs
    
    @torch.no_grad()
    def render_material_modalities(self, positions_subpixel, triangle_idx_subpixel, valid_subpixel, spp):
        outputs = Batch()
        modalities_to_render = self.render_cfg["modalities_to_render"]

        # Render BRDF modalities
        if any(m in modalities_to_render for m in self.BRDF_MODALITIES):
            # Query the BRDF
            mat_subpixel = self.brdf(position=positions_subpixel, triangle_idx=triangle_idx_subpixel, caching=False)
            mat_subpixel = Batch(mat_subpixel)

            mat_subpixel = mat_subpixel * valid_subpixel.unsqueeze(-1)

            # Average over spp
            mat_pixel = mat_subpixel.map(lambda x: einops.rearrange(x, "(b spp) c -> b spp c", spp=spp).mean(1))
            albedo_pixel, metallic_pixel, roughness_pixel = mat_pixel['albedo'],mat_pixel['metallic'],mat_pixel['roughness']

            outputs["albedo"] = albedo_pixel
            outputs["roughness"] = roughness_pixel
            outputs["metallic"] = metallic_pixel
        return outputs

    def save_renderings(self, outputs):
        for file_name, output_images in outputs.items():
            for output_name, output_image in output_images.items():
                # Prepare path
                extension = self.MODALITY_TO_EXTENSION[output_name]
                output_path = os.path.join(self.output["folder_path"], output_name, f"{file_name}{extension}")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # Post-processing
                if extension == ".png":
                    output_image = torch.clamp(output_image, 0.0, 1.0)

                # Save image
                save_image(output_image.permute(1,2,0).numpy(force=True), output_path)


@torch.no_grad()
def find_affine_transform(pred, target, mask=None):
    if mask is not None:
        pred = pred[mask.bool(), :]
        target = target[mask.bool(), :]

    N = pred.shape[0]
    ones = torch.ones((N, 1), dtype=pred.dtype, device=pred.device)
    P_aug = torch.cat([pred, ones], dim=1)  # (N, 4)

    # Solve least squares
    result = torch.linalg.lstsq(P_aug, target)
    A_t = result.solution             # (4, 3)
    A = A_t.T                         # (3, 4)
    return A


class TrajectoryRender(Render):
    def configure_model(self):
        super().configure_model()

        # ============================ Trajectory =============================
        self.trajectory_cfg = get_config(self.input["trajectory_cfg"])
        self.trajectory = hydra.utils.instantiate(self.input["trajectory_cfg"])

    @torch.no_grad()
    def run(self):
        # Seed
        torch.manual_seed(0)
        np.random.seed(0)

        # Prepare ray directions
        img_h, img_w = self.im_hw
        fov = self.render_cfg["fov"]
        if isinstance(fov, (int, float)):
            focal_x = focal_y = (0.5 * img_w / np.tan(0.5 * np.deg2rad(self.render_cfg["fov"]))).item()
        else:
            focal_x = (0.5 * img_w / np.tan(0.5 * np.deg2rad(fov[0]))).item()
            focal_y = (0.5 * img_h / np.tan(0.5 * np.deg2rad(fov[1]))).item()
        directions = get_ray_directions_focal(img_h, img_w, focal_x=focal_x, focal_y=focal_y)

        # Save the trajectory config
        self.save_trajectory()

        # Render and eval all images
        for image_idx, c2w in tqdm(enumerate(self.trajectory)):
            # Check if placeholder is present
            placeholder_path = os.path.join(self.output["folder_path"], "rgbs_ldr", f"{image_idx:04d}.png")
            if os.path.exists(placeholder_path):
                self.module_logger.info(f"Skipping {image_idx:04d} as it is already rendered ({placeholder_path}).")
                continue
            
            # Prepare rays for the current camera pose
            c2w = torch.FloatTensor(c2w[:3, :4])
            rays_o, rays_d, dxdu, dydv= get_rays(directions, c2w, focal=focal_x)  # h*w x 3
            rays = torch.cat([rays_o, rays_d, dxdu, dydv], -1).permute(2,0,1)

            # Render all the requested modalities
            rendered_outputs = self.render_single(rays=rays.to(self.device), 
                                                  exposure=self.render_cfg["exposure"],
                                                  file_name=f"{image_idx:04d}")

            # Save the renderings
            self.save_renderings(rendered_outputs)
        
        # Create a video from the rendered images
        for modality in self.render_cfg["modalities_to_render"]:
            extension = self.MODALITY_TO_EXTENSION[modality]
            output_folder = os.path.join(self.output["folder_path"], modality)
            video_path = os.path.join(self.output["folder_path"], f"{modality}.mp4")
            temp_video_path = os.path.join(self.output["folder_path"], f"{modality}_temp.mp4")
            os.system(f"ffmpeg -y -framerate {self.render_cfg.get('video_framerate',30)} -i {output_folder}/%04d{extension} -c:v libx264 -pix_fmt yuv420p {temp_video_path}")
            os.system(f"ffmpeg -y -i {temp_video_path} -filter_complex \"[0:v]reverse[r];[0:v][r]concat,format=yuv420p[v]\" -map \"[v]\" {video_path}")
            os.remove(temp_video_path)

        # Create a grid video if specified
        if self.render_cfg.get("grid_video", None) is not None:
            grid_video_path = os.path.join(self.output["folder_path"], "grid_video.mp4")
            input_videos = [os.path.join(self.output["folder_path"], f"{modality}.mp4") for modality in self.render_cfg["grid_video"]["modalities"]]
            self.create_grid_video(grid_video_path, input_videos, self.render_cfg["grid_video"]["rows"], self.render_cfg["grid_video"]["cols"])

    def create_grid_video(self, grid_video_path, video_files, num_rows, num_cols):
        # Load video clips
        clips = [VideoFileClip(video_file) for video_file in video_files]

        # Arrange clips in a grid
        grid = clips_array([[clips[row * num_cols + col] for col in range(num_cols)] for row in range(num_rows)])

        # Resize to full HD
        final_video = grid

        # Save the final video
        final_video.write_videofile(grid_video_path)

    def save_trajectory(self):
        trajectory_output_path = os.path.join(self.output["folder_path"], "trajectory.json")
        self.module_logger.info(f"Saving trajectory to {trajectory_output_path}...")
        transforms = dict()

        for image_idx, c2w in tqdm(enumerate(self.trajectory)):
            transforms[f"{image_idx:04d}"] = c2w.tolist()

        with open(trajectory_output_path, 'w') as f:
            json.dump(transforms, f, indent=4)

class RelightTrajectoryRender(Render):
    def configure_model(self):
        super().configure_model()

        # ============================ Trajectory =============================
        self.trajectory_cfg = get_config(self.input["trajectory_cfg"])
        self.trajectory = hydra.utils.instantiate(self.input["trajectory_cfg"])

        # ============================ Relighting =============================
        self.relighting_cfg = get_config(self.input["relighting_cfg"])
        self.relighting = hydra.utils.instantiate(self.input["relighting_cfg"])

    @torch.no_grad()
    def run(self):
        # Seed
        torch.manual_seed(0)
        np.random.seed(0)

        # Prepare ray directions
        img_h, img_w = self.im_hw
        focal = (0.5 * img_w / np.tan(0.5 * np.deg2rad(self.render_cfg["fov"]))).item()
        directions = get_ray_directions(img_h, img_w, focal)

        # Render and eval all images
        for image_idx, c2w in tqdm(enumerate(self.trajectory)):
            # Prepare rays for the current camera pose
            c2w = torch.FloatTensor(c2w[:3, :4])
            rays_o, rays_d, dxdu, dydv = get_rays(directions, c2w, focal=focal)  # h*w x 3
            rays = torch.cat([rays_o, rays_d, dxdu, dydv], -1).permute(2,0,1)

            # Render all the requested modalities
            rendered_outputs = self.render_single(rays=rays.to(self.device), 
                                                  exposure=self.render_cfg["exposure"],
                                                  file_name=f"{image_idx:04d}")

            # Save the renderings
            self.save_renderings(rendered_outputs)
        
        # Create a video from the rendered images
        for modality in self.render_cfg["modalities_to_render"]:
            extension = self.MODALITY_TO_EXTENSION[modality]
            output_folder = os.path.join(self.output["folder_path"], modality)
            video_path = os.path.join(self.output["folder_path"], f"{modality}.mp4")
            temp_video_path = os.path.join(self.output["folder_path"], f"{modality}_temp.mp4")
            os.system(f"ffmpeg -y -framerate {self.render_cfg.get('video_framerate',30)} -i {output_folder}/%04d{extension} -c:v libx264 -pix_fmt yuv420p {temp_video_path}")
            os.system(f"ffmpeg -y -i {temp_video_path} -filter_complex \"[0:v]reverse[r];[0:v][r]concat,format=yuv420p[v]\" -map \"[v]\" {video_path}")
            os.remove(temp_video_path)

        # Create a grid video if specified
        if self.render_cfg.get("grid_video", None) is not None:
            grid_video_path = os.path.join(self.output["folder_path"], "grid_video.mp4")
            input_videos = [os.path.join(self.output["folder_path"], f"{modality}.mp4") for modality in self.render_cfg["grid_video"]["modalities"]]
            self.create_grid_video(grid_video_path, input_videos, self.render_cfg["grid_video"]["rows"], self.render_cfg["grid_video"]["cols"])

    def create_grid_video(self, grid_video_path, video_files, num_rows, num_cols):
        # Load video clips
        clips = [VideoFileClip(video_file) for video_file in video_files]

        # Arrange clips in a grid
        grid = clips_array([[clips[row * num_cols + col] for col in range(num_cols)] for row in range(num_rows)])

        # Resize to full HD
        final_video = grid

        # Save the final video
        final_video.write_videofile(grid_video_path)
                






class MitsubaRender(Task):
    TASK_NAME = "render"
    MODALITY_TO_EXTENSION = {
        "rgbs_ldr": ".png",
        "rgbs_hdr": ".exr",
        "albedo": ".png",
        "albedo_scaled": ".png",
        "albedo_perfect_scaled": ".png",
        "roughness_perfect_scaled": ".png",
        "metallic_perfect_scaled": ".png",
        "roughness": ".png",
        "metallic": ".png",
        "emission": ".exr",
        "emission_mask": ".png",
    }
    BRDF_MODALITIES = ["albedo", "albedo_scaled", "roughness", "metallic"]
    IMAGE_MODALITIES = ["rgbs_ldr", "rgbs_hdr"]

    def __init__(self,
                 input,
                 output,
                 render_cfg,
                 **kwargs):
        super().__init__()
        
        self.input = input
        self.output = output
        self.render_cfg = render_cfg

        self.module_logger = init_logger()

        self.configure_model()

    def configure_model(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ============================ Mesh =============================
        # Initialize the scene
        assert os.path.exists(self.input["scene_path"]), 'Mesh not found: '+ self.input["scene_path"]

        # Load the dataset
        if self.input.get("dataset_cfg", None):
            self.dataset = hydra.utils.instantiate(self.input["dataset_cfg"])
            self.im_hw = self.dataset.resolution
        else:
            self.im_hw = (self.render_cfg["resolution"]["height"], self.render_cfg["resolution"]["width"])

        # ============================ Models =============================
        # Load the BRDF
        self.brdf_cfg = self.input["brdf"]["cfg"]
        self.brdf_pt = self.input["brdf"]["pt"]

        # Load the CRF
        self.crf_cfg = get_config(self.input["crf"])
        if self.crf_cfg is None:
            self.crf = None
        else:
            self.crf = hydra.utils.instantiate(self.input["crf"])

        # Load the Emitter
        self.emitter_cfg = self.input["emitter"]["cfg"]
        self.emitter_pt = self.input["emitter"]["pt"]

        # =========================== Render =============================
        # Set up a denoiser
        self.denoiser = mitsuba.OptixDenoiser([self.im_hw[1], self.im_hw[0]])

        # ============================ Trajectory =============================
        self.trajectory_cfg = get_config(self.input["trajectory_cfg"])
        self.trajectory = hydra.utils.instantiate(self.input["trajectory_cfg"])

        # ============================ Relighting =============================
        self.relighting_cfg = get_config(self.input["relighting_cfg"])
        self.relighting = hydra.utils.instantiate(self.input["relighting_cfg"]["interpolation"])

        # ============================ Mesh =============================
        self.scene = self.prepare_scene()

    def log_config(self, cfg):
        # Implement logging logic here
        super().log_config(cfg)

    @torch.no_grad()
    def run(self):
        # Seed
        torch.manual_seed(0)
        np.random.seed(0)

        # Render and eval all images
        for image_idx, c2w in tqdm(enumerate(self.trajectory)):
            # Prepare rays for the current camera pose
            camera_transform = self.get_camera_transform(c2w)

            # Check if placeholder is present
            placeholder_path = os.path.join(self.output["folder_path"], "rgbs_ldr", f"{image_idx:04d}.png")
            if os.path.exists(placeholder_path):
                self.module_logger.info(f"Skipping {image_idx:04d} as it is already rendered ({placeholder_path}).")
                continue

            os.makedirs(os.path.dirname(placeholder_path), exist_ok=True)
            with open(placeholder_path, 'w') as f:
                f.write('In progress')

            # Render all the requested modalities
            rendered_outputs = self.render_single(camera_transform=camera_transform,
                                                  image_idx=image_idx,
                                                  exposure=self.render_cfg["exposure"],
                                                  file_name=f"{image_idx:04d}")

            # Save the renderings
            self.save_renderings(rendered_outputs)
        
        # Create a video from the rendered images
        for modality in self.render_cfg["modalities_to_render"]:
            extension = self.MODALITY_TO_EXTENSION[modality]
            output_folder = os.path.join(self.output["folder_path"], modality)
            video_path = os.path.join(self.output["folder_path"], f"{modality}.mp4")
            temp_video_path = os.path.join(self.output["folder_path"], f"{modality}_temp.mp4")
            os.system(f"ffmpeg -y -framerate {self.render_cfg.get('video_framerate',30)} -i {output_folder}/%04d{extension} -c:v libx264 -pix_fmt yuv420p {temp_video_path}")
            os.system(f"ffmpeg -y -i {temp_video_path} -filter_complex \"[0:v]reverse[r];[0:v][r]concat,format=yuv420p[v]\" -map \"[v]\" {video_path}")
            os.remove(temp_video_path)

        # Create a grid video if specified
        if self.render_cfg.get("grid_video", None) is not None:
            grid_video_path = os.path.join(self.output["folder_path"], "grid_video.mp4")
            input_videos = [os.path.join(self.output["folder_path"], f"{modality}.mp4") for modality in self.render_cfg["grid_video"]["modalities"]]
            self.create_grid_video(grid_video_path, input_videos, self.render_cfg["grid_video"]["rows"], self.render_cfg["grid_video"]["cols"])

    def create_grid_video(self, grid_video_path, video_files, num_rows, num_cols):
        # Load video clips
        clips = [VideoFileClip(video_file) for video_file in video_files]

        # Arrange clips in a grid
        grid = clips_array([[clips[row * num_cols + col] for col in range(num_cols)] for row in range(num_rows)])

        # Resize to full HD
        final_video = grid

        # Save the final video
        final_video.write_videofile(grid_video_path)

    @torch.no_grad()
    def render_single(self, 
                      camera_transform,
                      image_idx,
                      exposure,
                      file_name):
        # self.module_logger.info(f"Rendering {batch}...")
        outputs = Batch()
        modalities_to_render = self.render_cfg["modalities_to_render"]

        # Update the scene params
        params = mitsuba.traverse(self.scene)
        params['camera.to_world'] = camera_transform

        for lighting_name, lighting_transform in self.relighting.items():
            params[f'{lighting_name}.to_world'] = lighting_transform[image_idx]
        params.update()

        # Get kwargs
        SPP = self.render_cfg["spp"]
        spp = self.render_cfg["spp_batch"]
        assert SPP % spp == 0, f"spp should be divisible by spp_batch, but got {SPP} and {spp}"

        # =================== Render Image ==================
        L = torch.zeros(*self.im_hw, 3)
        seed = 0
        for _ in range(SPP//spp):
            # render color with path tracing
            L += torch.nan_to_num(mitsuba.render(self.scene,spp=spp,seed=seed).torch().cpu(), nan=0.)
            seed += 1

        L = L / (SPP//spp)
        rgbs_hdr = L 

        if self.render_cfg["denoise"]:
            rgbs_hdr = self.denoiser(rgbs_hdr.numpy(force=True)).torch().to(rgbs_hdr.device)
            

        if "rgbs_hdr" in modalities_to_render:
            outputs["rgbs_hdr"] = rgbs_hdr.permute(2,0,1)

        if "rgbs_ldr" in modalities_to_render:
            # Apply the camera response function
            rgbs_hdr = einops.rearrange(rgbs_hdr, 'h w c -> (h w) c')
            if isinstance(exposure, (int, float)):
                exposure = torch.ones(len(rgbs_hdr), 1) * exposure
            else:
                exposure = einops.rearrange(exposure, '... c h w -> (h w) ... c')
            rgbs_ldr = self.crf(rgbs_hdr, exposure)
            rgbs_ldr = einops.rearrange(rgbs_ldr, '(h w) c -> h w c', h=self.im_hw[0], w=self.im_hw[1])
            outputs["rgbs_ldr"] = rgbs_ldr.permute(2,0,1)

        # =================== Reshape Outputs ==================
        outputs = Batch({
            file_name: outputs
        })
        return outputs
    
    def get_camera_transform(self, c2w):
        t = c2w[:3, 3]
        up = c2w[:3, 1]
        forward = c2w[:3, 2]

        to_world = mitsuba.ScalarTransform4f().look_at(origin=mitsuba.ScalarPoint3f(t), 
                                                       target=mitsuba.ScalarPoint3f(t+forward), 
                                                       up=mitsuba.ScalarPoint3f(up))
        return to_world
    
    def prepare_scene(self):
        from iif.component.rendering.pir_mitsuba import PIRBSDF

        scene_dict = {
            "type": "scene",

            "main_scene": {
                'type': os.path.splitext(self.input["scene_path"])[-1].replace('.', ''),
                'filename': self.input["scene_path"],
                'bsdf': {
                    'type': 'pir_bsdf',
                    'brdf_cfg': self.brdf_cfg,
                    'brdf_pt': self.brdf_pt,
                    'emitter_cfg': self.emitter_cfg,
                    'emitter_pt': self.emitter_pt
                }
                # 'bsdf': {
                #     'type': 'twosided',
                #     "pir_bsdf": {
                #         'type': 'pir_bsdf',
                #         'brdf_cfg': self.brdf_cfg,
                #         'brdf_pt': self.brdf_pt,
                #         'emitter_cfg': self.emitter_cfg,
                #         'emitter_pt': self.emitter_pt
                #     }
                # }
            },
            "integrator": {
                "type": "path",
                "max_depth": self.render_cfg["depth"]
            },

            # "lighting": self.input["relighting_cfg"]["lighting_cfg"],

            "camera": {
                "type": "perspective",
                "fov": self.render_cfg["fov"],
                "film": {
                    "type": "hdrfilm",
                    "width": self.im_hw[1],
                    "height": self.im_hw[0],
                    "rfilter": {
                        "type": "box"
                    }
                },
            }
        }
        scene_dict.update(self.input["relighting_cfg"]["lighting_cfg"])
        return mitsuba.load_dict(scene_dict)

    def save_renderings(self, outputs):
        for file_name, output_images in outputs.items():
            for output_name, output_image in output_images.items():
                # Prepare path
                extension = self.MODALITY_TO_EXTENSION[output_name]
                output_path = os.path.join(self.output["folder_path"], output_name, f"{file_name}{extension}")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # Post-processing
                if extension == ".png":
                    output_image = torch.clamp(output_image, 0.0, 1.0)

                # Save image
                save_image(output_image.permute(1,2,0).numpy(force=True), output_path)
