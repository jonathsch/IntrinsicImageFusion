import copy
import os
import einops
import hydra
import mitsuba
import numpy as np
from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch
import torch_scatter
import torch.nn.functional as F
from tqdm import tqdm
import trimesh

from iif.component.rendering.ops import lerp_specular
from iif.component.rendering.path_tracing import path_tracing, path_tracing_single, ray_intersect
from iif.utils.batching import batched_average, batched_eval
from iif.utils.datastructure import Batch
from iif.utils.image_io import save_image, show_image
from iif.utils.logging import init_logger
from iif.utils.model import get_config


class InverseRenderingModule(pl.LightningModule):
    def __init__(self, 
                 forward_cfg,
                 model_cfg, 
                 loss_cfg, 
                 optimizer_cfg,
                 scheduler_cfg,
                 *args, **kwargs):
        super(InverseRenderingModule, self).__init__(*args, **kwargs)
        self.module_logger = init_logger()
        
        self.forward_cfg = self.default_forward_cfg()
        self.forward_cfg.update(forward_cfg)

        self.model_cfg = model_cfg

        self.loss_cfg = self.default_loss_cfg()
        self.loss_cfg.update(loss_cfg)

        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

        self.configure_model()

        self.loss_fn = F.mse_loss
    
    def default_forward_cfg(self):
        return {
            "jitter_within_pixel": False,
            "spp": 1,
            "spp_batch": 1,
            "depth": 1,
            "grad_depth": None,
            "rendering_type": "path_tracing",
            "denoising": False,
            "do_prefilter": True,
            "do_emissionfilter": False
        }
    
    def default_loss_cfg(self):
        return {
            "weight": dict(),
            "rgb_type": "ldr",  # ldr | hdr
            "regularizations": []
        }

    def configure_model(self):
        # Initialize the scene - TODO: Consider delegating it to a spearate datamodule
        assert os.path.exists(self.model_cfg["scene_path"]), 'Mesh not found: '+ self.model_cfg["scene_path"]
        self.scene = mitsuba.load_dict({
            'type': 'scene',
            'shape_id':{
                'type': os.path.splitext(self.model_cfg["scene_path"])[-1].replace('.', ''),
                'filename': self.model_cfg["scene_path"]
            }
        })

        # Load the BRDF
        self.brdf_cfg = get_config(self.model_cfg["brdf"])
        self.brdf = hydra.utils.instantiate(self.model_cfg["brdf"])

        # Load the CRF
        self.crf_cfg = get_config(self.model_cfg["crf"])
        self.crf = hydra.utils.instantiate(self.model_cfg["crf"])

        # Load the Emitter
        self.emitter_cfg = get_config(self.model_cfg["emitter"])
        self.emitter = hydra.utils.instantiate(self.model_cfg["emitter"])

    def log_config(self, cfg):
        out_folder = cfg["output"]["folder_path"]
        os.makedirs(out_folder, exist_ok=True)

        # Save BRDF config
        OmegaConf.save(self.brdf_cfg, os.path.join(out_folder, f"brdf.yaml"))

        # Save CRF config
        OmegaConf.save(self.crf_cfg, os.path.join(out_folder, f"crf.yaml"))

        # Save Emitter config
        OmegaConf.save(self.emitter_cfg, os.path.join(out_folder, f"emitter.yaml"))

        self.module_logger.info(f"Configs saved to {out_folder}")

    def reinit(self):
        raise NotImplementedError("Reinit not implemented.")
        pass

    def forward(self, x):
        raise NotImplementedError("Forward method should be implemented in the subclass.")
    
    def step(self, batch, batch_idx=None, **kwargs):
        # Get kwargs
        forward_cfg = copy.deepcopy(self.forward_cfg)
        forward_cfg.update(kwargs)

        SPP = forward_cfg["spp"]
        spp = forward_cfg["spp_batch"]

        # Extract data from the batch
        rays = batch['rays']
        xs_pixel,ds_pixel = rays[:,:3], rays[:,3:6]
        ds_pixel = F.normalize(ds_pixel,dim=1)
        dxdu_pixel,dydv_pixel = rays[:,6:9],rays[:,9:12]

        # =================== Ray-Scene Intersection ==================
        # Intersect with pixel center for pre-filtering
        positions_pixel, normals_pixel, _, triangle_idx_pixel, valid_pixel = ray_intersect(self.scene,xs_pixel,ds_pixel)

        if not valid_pixel.any():
            return None
        
        exposure_pixel = batch['exposure']
        if forward_cfg["do_prefilter"]:
            exposure_pixel = exposure_pixel[valid_pixel]
            positions_pixel = positions_pixel[valid_pixel]
            normals_pixel = normals_pixel[valid_pixel]
            triangle_idx_pixel = triangle_idx_pixel[valid_pixel]
            xs_pixel, ds_pixel = xs_pixel[valid_pixel], ds_pixel[valid_pixel]
            dxdu_pixel, dydv_pixel = dxdu_pixel[valid_pixel], dydv_pixel[valid_pixel]

        # assert not (not forward_cfg["jitter_within_pixel"] and SPP > 1), f"Jittering within pixel is not allowed, but spp is higher than 1"
        if forward_cfg["jitter_within_pixel"]:
            # Find the intersection of rays with the scene
            xs_subpixel = xs_pixel.unsqueeze(1).repeat(1, SPP, 1).view(-1, xs_pixel.shape[1])
            ds_subpixel = ds_pixel.unsqueeze(1).repeat(1, SPP, 1).view(-1, ds_pixel.shape[1])
            dxdu_subpixel = dxdu_pixel.unsqueeze(1).repeat(1, SPP, 1).view(-1, dxdu_pixel.shape[1])
            dydv_subpixel = dydv_pixel.unsqueeze(1).repeat(1, SPP, 1).view(-1, dydv_pixel.shape[1])

            # Sample within pixel
            du,dv = torch.rand(2, len(xs_subpixel), 1, device=xs_subpixel.device) - 0.5
            ds_subpixel = F.normalize(ds_subpixel + dxdu_subpixel * du + dydv_subpixel * dv, dim=1)

            positions_subpixel, normals_subpixel, _, triangle_idx_subpixel, valid_subpixel = batched_eval(ray_intersect, Batch(xs=einops.rearrange(xs_subpixel, "(b spp) ... -> b spp ...", spp=SPP), 
                                                                                                                               ds=einops.rearrange(ds_subpixel, "(b spp) ... -> b spp ...", spp=SPP)), 
                                                                                                                        SPP, 
                                                                                                                        spp,
                                                                                                                        scene=self.scene)
            positions_subpixel = einops.rearrange(positions_subpixel, "b spp ... -> (b spp) ...")
            normals_subpixel = einops.rearrange(normals_subpixel, "b spp ... -> (b spp) ...")
            triangle_idx_subpixel = einops.rearrange(triangle_idx_subpixel, "b spp ... -> (b spp) ...")
            valid_subpixel = einops.rearrange(valid_subpixel, "b spp ... -> (b spp) ...")

            # positions_subpixel, normals_subpixel, _, triangle_idx_subpixel, valid_subpixel = ray_intersect(self.scene,xs_subpixel,ds_subpixel)

            valid_pixel = einops.rearrange(valid_subpixel, "(b spp) -> b spp", spp=SPP).all(1)
            if not forward_cfg["do_prefilter"]:
                valid_pixel = torch.ones_like(valid_pixel)
            elif not valid_pixel.any():
                return None
            valid_subpixel = valid_pixel.repeat(1, SPP).view(-1)
            
            exposure_pixel = exposure_pixel[valid_pixel]
            positions_subpixel = positions_subpixel[valid_subpixel]
            normals_subpixel = normals_subpixel[valid_subpixel]
            triangle_idx_subpixel = triangle_idx_subpixel[valid_subpixel]
            xs_subpixel, ds_subpixel = xs_subpixel[valid_subpixel], ds_subpixel[valid_subpixel]
            dxdu_subpixel, dydv_subpixel = dxdu_subpixel[valid_subpixel], dydv_subpixel[valid_subpixel]
        else:
            positions_subpixel = positions_pixel
            ds_subpixel = ds_pixel
            normals_subpixel = normals_pixel
            triangle_idx_subpixel = triangle_idx_pixel
            valid_subpixel = valid_pixel

        # ================ Rendering =================
        if forward_cfg["rendering_type"] == "path_tracing":
            # Rendering with Path Tracing
            L = torch.zeros_like(xs_pixel)
            for _ in range(SPP//spp):
                L += path_tracing(
                    self.scene, self.emitter, self.brdf, 
                    xs_pixel, ds_pixel, dxdu_pixel, dydv_pixel, spp,
                    depth=forward_cfg["depth"],
                    grad_depth=forward_cfg["grad_depth"]
                )
            L = L / (SPP//spp)
            rgbs_hdr = L 

        elif forward_cfg["rendering_type"] == "shading_cache":
            # Rendering with pre-calculated shading cache
            # Collect the cache
            diffuse_pixel = batch['diffuse_shading_cache']
            specular0_pixel = batch['specular0_shading_cache']
            specular1_pixel = batch['specular1_shading_cache']

            if forward_cfg["do_prefilter"]:
                diffuse_pixel = diffuse_pixel[valid_pixel]
                specular0_pixel = specular0_pixel[valid_pixel]
                specular1_pixel = specular1_pixel[valid_pixel]

            # Query the BRDF
            mat_pixel = batched_average(self.brdf, Batch(position=einops.rearrange(positions_subpixel, "(b spp) ... -> b spp ...", spp=SPP),
                                                         triangle_idx=einops.rearrange(triangle_idx_subpixel, "(b spp) ... -> b spp ...", spp=SPP))
                                                         , SPP, spp)
            # mat_subpixel = self.brdf(positions_subpixel)
            # # Average over spp
            # mat_subpixel = Batch(mat_subpixel)
            # mat_pixel = mat_subpixel.map(lambda x: einops.rearrange(x, "(b spp) c -> b spp c", spp=SPP).mean(1))
            albedo_pixel, metallic_pixel, roughness_pixel = mat_pixel['albedo'],mat_pixel['metallic'],mat_pixel['roughness']

             # Diffuse and specular reflectance
            kd_pixel = albedo_pixel*(1-metallic_pixel)
            ks_pixel = 0.04*(1-metallic_pixel) + albedo_pixel*metallic_pixel

            # Diffuse component and specular component
            Ld_pixel = kd_pixel*diffuse_pixel
            Ls_pixel = ks_pixel*lerp_specular(specular0_pixel,roughness_pixel)+lerp_specular(specular1_pixel,roughness_pixel)

            # Emission
            Le_pixel, _, _, emission_mask_pixel = batched_average(self.emitter.eval_emitter, 
                                                                        Batch(position=einops.rearrange(positions_subpixel, "(b spp) ... -> b spp ...", spp=SPP), 
                                                                              light_dir=einops.rearrange(ds_subpixel, "(b spp) ... -> b spp ...", spp=SPP), 
                                                                              triangle_idx=einops.rearrange(triangle_idx_subpixel, "(b spp) ... -> b spp ...", spp=SPP)),
                                                                        SPP, spp)
            emission_mask_pixel = emission_mask_pixel > 0.
            # Le_subpixel, _, _, emission_mask_subpixel = self.emitter.eval_emitter(positions_subpixel, ds_subpixel, triangle_idx_subpixel)
            # emission_mask_pixel = einops.rearrange(emission_mask_subpixel, "(b spp) -> b spp", spp=SPP).any(1)    
            # Le_pixel = einops.rearrange(Le_subpixel, "(b spp) c -> b spp c", spp=SPP).mean(1)

            # Final color
            L_pixel = Ld_pixel + Ls_pixel + Le_pixel
            rgbs_hdr = L_pixel

            # Mask out the emission on non-emissive surfaces
            if forward_cfg["do_emissionfilter"]:
                valid_pixel[valid_pixel.clone()] &= (~emission_mask_pixel)
                rgbs_hdr = rgbs_hdr[~emission_mask_pixel]
                exposure_pixel = exposure_pixel[~emission_mask_pixel]
        else:
            raise NotImplementedError(f"Rendering type {forward_cfg['rendering_type']} not implemented.")
        
        # =============== Denoising =================
        if forward_cfg["denoising"]:
            B = kwargs.get("B", None)
            H = kwargs.get("H", None)
            W = kwargs.get("W", None)
            rgbs_hdr = einops.rearrange(rgbs_hdr, '(b h w) c -> b h w c', b=B, h=H, w=W)

            denoiser = mitsuba.OptixDenoiser([W,H])
            # self.module_logger.info(f"Denoising the rendered images: {rgbs_hdr}")
            rgbs_hdr = torch.stack([denoiser(rgbs.numpy(force=True)).torch().to(rgbs_hdr.device) for rgbs in rgbs_hdr], dim=0)

            rgbs_hdr = einops.rearrange(rgbs_hdr, 'b h w c -> (b h w) c')

        # =============== CRF =================

        rgbs_ldr = self.crf(rgbs_hdr, exposure_pixel)

        # =============== Output =================
        # Compose output
        output = Batch(
            rgbs_hdr=rgbs_hdr,
            rgbs_ldr=rgbs_ldr,
            positions=positions_subpixel,
            directions=ds_subpixel,
            normals=normals_subpixel,
            triangle_idx=triangle_idx_subpixel,
            valid=valid_pixel,
            spp=SPP,
            spp_batch=spp,
        )

        return output

    def training_step(self, batch, batch_idx=None, **kwargs):
        # ================ Data =================
        # Reshape to ray-based
        if batch['rays'].ndim > 3:
            # Assume image-based batch
            ray_based_batch = False
            B, _, H, W = batch['rays'].shape
            batch = batch.map(lambda x: einops.rearrange(x, 'b ... c h w -> (b h w) ... c'))
        else:
            # Assume ray-based batch
            ray_based_batch = True
            batch = batch.map(lambda x: einops.rearrange(x, 'b r ... c -> (b r) ... c'))

        # ================ Step =================
        output = self.step(batch, batch_idx, **kwargs)

        # ================ Loss Calculation =================

        # Collect all losses
        loss_info = Batch()

        # Rendering loss
        valid = output['valid']
        if self.loss_cfg["rgb_type"] == "ldr":
            rgbs_gt = batch['rgbs_ldr'][valid]
            rgbs_pred = output['rgbs_ldr']
        else:
            rgbs_gt = batch['rgbs_hdr'][valid]
            rgbs_pred = output['rgbs_hdr']

        rgbs_pred = rgbs_pred.clamp(0, 0.9)
        rgbs_gt = rgbs_gt.clamp(0, 0.9)
        loss_info["render"] = self.loss_fn(rgbs_pred, rgbs_gt)

        # Add regularizations
        if "brdf" in self.loss_cfg["regularizations"]:
            loss_info["brdf"] = Batch(self.brdf.get_regularization_loss())

        if "crf" in self.loss_cfg["regularizations"]:
            loss_info["crf"] = Batch(self.crf.get_regularization_loss())
        
        if "emitter" in self.loss_cfg["regularizations"]:
            loss_info["emitter"] = Batch(self.emitter.get_regularization_loss())

        # Compose the loss
        loss_info = loss_info.flatten(separator="_")
        weight = Batch(dict(self.loss_cfg["weight"]), default=lambda: 1.0)
        loss = sum(list((loss_info * weight).values()))

        if loss.requires_grad == False:
            self.module_logger.warning("Loss has no grad! Skipping this step.")
            return None

        # ================ Metrics =================

        psnr = -10.0 * torch.log10(loss_info["render"].clamp_min(1e-5))

        # =================== Output ==================

        output = {
            'loss': loss,
            'metric/psnr': psnr
        }

        loss_info = loss_info.map_keys(lambda x: f'loss/{x}')
        output.update(loss_info.to_dict())

        return output
    
    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        # Run backward pass first (Lightning handles this before calling optimizer_step)
        closure_result = closure()

        # Check gradients for NaNs or Infs
        has_nan = False
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        has_nan = True
                        break
            if has_nan:
                break

        if has_nan:
            print(f"⚠️  Skipping optimizer step at batch {batch_idx} due to NaN/Inf gradients.")
            optimizer.zero_grad()
            return  # Skip this update safely

        # Otherwise, take the step normally
        optimizer.step(closure=closure)

    def validation(self, batch, batch_idx=None, **kwargs):
        # ================ Data =================
        # Assume image-based batch
        assert batch['rays'].ndim > 3, "Batch should be image-based with shape (B, C, H, W) or ray-based with shape (B, R, C)"
        B, _, H, W = batch['rays'].shape
        batch = batch.map(lambda x: einops.rearrange(x, 'b ... c h w -> (b h w) ... c'))

        # ================ Step =================
        validation_output = Batch(default=Batch)
        step_output = self.step(batch, batch_idx, B=B, H=H, W=W, **kwargs)

        rgbs_ldr = step_output['rgbs_ldr']
        rgbs_hdr = step_output['rgbs_hdr']
        positions = step_output['positions']
        directions = step_output['directions']
        triangle_idx = step_output['triangle_idx']
        SPP = step_output['spp']
        spp = step_output['spp_batch']

        # ================= Metrics =================
        metrics = Batch(default=Batch)

        # RGB Metrics
        if "rgbs_ldr" in batch:
            rgbs_gt = batch['rgbs_ldr']
            metrics["rgb_ldr"]["l2"] = F.mse_loss(rgbs_ldr, rgbs_gt)
            metrics["rgb_ldr"]["psnr"] = -10.0 * torch.log10(metrics["rgb_ldr"]["l2"].clamp_min(1e-20))
        if "rgbs_hdr" in batch:
            rgbs_gt = batch['rgbs_hdr']
            metrics["rgb_hdr"]["l2"] = F.mse_loss(rgbs_hdr, rgbs_gt)
            metrics["rgb_hdr"]["psnr"] = -10.0 * torch.log10(metrics["rgb_hdr"]["l2"].clamp_min(1e-20))

        # Emitter Metrics
        emission_pred, _, _, _ = batched_average(self.emitter.eval_emitter, 
                                                                        Batch(position=einops.rearrange(positions, "(b spp) ... -> b spp ...", spp=SPP), 
                                                                              light_dir=einops.rearrange(directions, "(b spp) ... -> b spp ...", spp=SPP), 
                                                                              triangle_idx=einops.rearrange(triangle_idx, "(b spp) ... -> b spp ...", spp=SPP)),
                                                                        SPP, spp)
        # emission_pred,_,_,_ = self.emitter.eval_emitter(step_output["positions"],None,step_output["triangle_idx"])
        # emission_pred = einops.rearrange(emission_pred, "(b spp) c -> b spp c", spp=spp).mean(1)
        emission_pred_mask = emission_pred.sum(-1) > 0
        emission_mask = emission_pred_mask
        if "emission" in batch:
            emission_gt = batch['emission']
            emission_gt_mask = emission_gt.sum(-1) > 0
            emission_mask = emission_gt_mask
            metrics["emitter"]["emission/l2"] = F.mse_loss(emission_pred, emission_gt)
            metrics["emitter"]["emission/log_psnr"] = -10.0 * torch.log10(metrics["emitter"]["emission/l2"].clamp_min(1e-20))
            metrics["emitter"]["emission/log_l2"] = F.mse_loss(torch.log(emission_pred + 1), torch.log(emission_gt + 1))
            metrics["emitter"]["emission/log_psnr"] = -10.0 * torch.log10(metrics["emitter"]["emission/log_l2"].clamp_min(1e-20))
            metrics["emitter"]["emission/iou"] = (emission_gt_mask & emission_pred_mask).sum() / (emission_gt_mask | emission_pred_mask).sum()

        # BRDF Metrics
        mat = batched_average(self.brdf, Batch(position=einops.rearrange(positions, "(b spp) ... -> b spp ...", spp=SPP),
                                               triangle_idx=einops.rearrange(triangle_idx, "(b spp) ... -> b spp ...", spp=SPP)),
                                               SPP, spp)
        # mat = self.brdf(step_output["positions"])
        # mat = Batch(mat)
        # mat = mat.map(lambda x: einops.rearrange(x, "(b spp) c -> b spp c", spp=spp).mean(1))

        # Roughness Metrics
        roughness_pred = mat['roughness']
        roughness_mask = roughness_pred == 1
        if 'roughness' in batch:
            roughness_gt = batch['roughness']

            roughness_gt[emission_mask] = 0
            roughness_pred[emission_mask] = 0

            metrics["brdf"]["roughness/l2"] = F.mse_loss(roughness_pred, roughness_gt)
            metrics["brdf"]["roughness/psnr"] = -10.0 * torch.log10(metrics["brdf"]["roughness/l2"].clamp_min(1e-20))

            roughness_mask = roughness_gt == 1
        roughness_mask = roughness_mask.squeeze(1)

        # Metallic Metrics
        metallic_pred = mat['metallic']
        if 'metallic' in batch:
            metallic_gt = batch['metallic']

            metallic_gt[emission_mask] = 0
            metallic_pred[emission_mask] = 0

            metrics["brdf"]["metallic/l2"] = F.mse_loss(metallic_pred, metallic_gt)
            metrics["brdf"]["metallic/psnr"] = -10.0 * torch.log10(metrics["brdf"]["metallic/l2"].clamp_min(1e-20))

        # Albedo Metrics
        if 'albedo' in batch:
            albedo_gt = batch['albedo']
            albedo_pred = mat['albedo']

            albedo_gt[emission_mask] = 0
            albedo_pred[emission_mask] = 0

            # albedo_gt[~roughness_mask] = 0
            # albedo_pred[~roughness_mask] = 0

            # albedo_pred = albedo_pred * (1 - metallic_pred)

            metrics["brdf"]["albedo/l2"] = F.mse_loss(albedo_pred, albedo_gt)
            metrics["brdf"]["albedo/psnr"] = -10.0 * torch.log10(metrics["brdf"]["albedo/l2"].clamp_min(1e-20))

            # Get scale invariant albedos as well
            # Transform the materials to the ground truth
            def affine_transform_torch(P, Q, mask=None):
                if mask is not None:
                    P = P[mask.bool(), :]
                    Q = Q[mask.bool(), :]

                # Solve least squares
                result = torch.linalg.lstsq(P, Q)
                A_t = result.solution             # (4, 3)
                A = A_t.T                         # (3, 4)
                return A
            
            def scale_transform_torch(P, Q, mask=None):
                if mask is not None:
                    P = P[mask.bool(), :]
                    Q = Q[mask.bool(), :]

                Q_mean = Q.mean(dim=0, keepdim=True)
                P_mean = P.mean(dim=0, keepdim=True)

                # Transform to align the means
                scale = Q_mean / P_mean.clamp(1e-20)
                A = torch.eye(3).to(scale) * scale
                return A
            
            def fit_and_transform(pred, target, mask=None):
                # transform = affine_transform_torch(pred, target, mask)
                transform = scale_transform_torch(pred, target, mask)
                return einops.einsum(pred, transform, "B D, C D -> B C").clamp(0,1).nan_to_num(0)
            
            albedo_pred_si = fit_and_transform(albedo_pred, albedo_gt, mask=~emission_mask)
            albedo_pred_si[emission_mask] = 0

            metrics["brdf"]["albedo_si/l2"] = F.mse_loss(albedo_pred_si, albedo_gt)
            metrics["brdf"]["albedo_si/psnr"] = -10.0 * torch.log10(metrics["brdf"]["albedo_si/l2"].clamp_min(1e-20))
        else: 
            albedo_pred_si = None

        # CRF Metrics
        if 'crf' in batch:
            crf_gt = batch['crf'][0]
            crf_pred = self.crf.get_crf()
            metrics["crf"]["crf/l2"] = F.mse_loss(crf_pred, crf_gt)

        validation_output["metrics"] = metrics.map(lambda x: x.unsqueeze(0))

        # ================ Qualitatives =================
        # Add RGB logs
        validation_output["rgb"]["render_ldr"] = einops.rearrange(rgbs_ldr, '(b h w) c -> b c h w', b=B, h=H, w=W)
        validation_output["rgb"]["render_hdr"] = einops.rearrange(rgbs_hdr, '(b h w) c -> b c h w', b=B, h=H, w=W)
        
        # Add BRDF logs
        if hasattr(self.brdf, 'log_details'):
            validation_output["brdf"] = self.brdf.log_details(position=positions, triangle_idx=triangle_idx, b=B, h=H, w=W, spp=SPP, spp_batch=spp, mask=~emission_mask)
        if albedo_pred_si is not None:
            validation_output["brdf"]["si/albedo"] = einops.rearrange(albedo_pred_si, '(b h w) c -> b c h w', b=B, h=H, w=W)

        # Add CRF logs
        if hasattr(self.crf, 'log_details'):
            validation_output["crf"] = self.crf.log_details(batch.get('crf', [None,])[0])

        # Add Emitter logs
        if hasattr(self.emitter, 'log_details'):
            validation_output["emitter"] = self.emitter.log_details(positions, directions, triangle_idx, b=B, h=H, w=W, spp=SPP, spp_batch=spp, emission_gt=batch.get('emission', None))

        return validation_output
    
    # def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):        
    #     set_to_none = False
    #     if self.optimizer_cfg.optimizer == 'MaskedAdam':
    #         set_to_none = True

    #     optimizer.zero_grad(set_to_none=set_to_none)

    def configure_optimizers(self):
        params = {n:p for n, p in self.named_parameters() if p.requires_grad}
        params_log = "\n".join([f'{n}: {p.numel()} - {p.shape}' for n, p in params.items()])
        self.module_logger.info(
            f"\n======== Optimizing {len(params)} Set of Variables =========\n" \
            f"{params_log}" \
            f"\n======================================"
        )

        if self.optimizer_cfg.optimizer == 'SGD':
            opt = torch.optim.SGD
            optimizer = opt(list(params.values()), lr=self.optimizer_cfg.learning_rate, weight_decay=self.optimizer_cfg.weight_decay)    
        elif self.optimizer_cfg.optimizer == 'Adam':
            opt = torch.optim.Adam
            optimizer = opt(list(params.values()), lr=self.optimizer_cfg.learning_rate, weight_decay=self.optimizer_cfg.weight_decay)    
        elif self.optimizer_cfg.optimizer == 'MaskedAdam':
            from iif.component.optimizer.masked_adam import MaskedAdam
            opt = MaskedAdam
            optimizer = opt(list(params.values()), lr=self.optimizer_cfg.learning_rate, weight_decay=self.optimizer_cfg.weight_decay)    
        else:
            raise NotImplementedError(f"Optimizer {self.optimizer_cfg.optimizer} not implemented.")
        
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=self.scheduler_cfg.milestones,gamma=self.scheduler_cfg.scheduler_rate)
        return [optimizer], [scheduler]
    