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

from iif.component.model.brdf import NGPBRDF, NGPAssignment, ObjTransformedBRDF
from iif.component.rendering.path_tracing import path_tracing_single, path_tracing_single_obj_mat, ray_intersect
from iif.utils.datastructure import Batch
from iif.utils.image_io import show_image
from iif.utils.logging import init_logger
from iif.utils.model import freeze_model


def create_color_map(num_classes, seed=42):
    """Generate a color map for num_classes IDs."""
    g = torch.Generator().manual_seed(seed)  # deterministic colors
    colors = torch.randint(0, 256, (num_classes, 3), dtype=torch.uint8, generator=g) / 255
    return colors


class EmitterModule(pl.LightningModule):
    def __init__(self, 
                 forward_cfg,
                 model_cfg, 
                 loss_cfg, 
                 optimizer_cfg,
                 scheduler_cfg,
                 *args, **kwargs):
        super(EmitterModule, self).__init__(*args, **kwargs)
        self.module_logger = init_logger()
        
        self.forward_cfg = self.default_forward_cfg()
        self.forward_cfg.update(forward_cfg)

        self.model_cfg = model_cfg

        self.loss_cfg = self.default_loss_cfg()
        self.loss_cfg.update(loss_cfg)

        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

        self.configure_model()
        self.register_buffer('color_map', create_color_map(self.model_cfg.num_objects))

        self.loss_fn = F.mse_loss
    
    def default_forward_cfg(self):
        return {
            "use_transform": False,
            "jitter_within_pixel": False,
            "spp": 1,
            "spp_batch": 1
        }
    
    def default_loss_cfg(self):
        return {
            "w_albedo_transform": 1e+2
        }

    def configure_model(self):
        # Initialize the scene - TODO: Consider delegating it to a spearate datamodule
        assert os.path.exists(self.model_cfg.scene_path), 'Mesh not found: '+ self.model_cfg.scene_path
        self.scene = mitsuba.load_dict({
            'type': 'scene',
            'shape_id':{
                'type': os.path.splitext(self.model_cfg.scene_path)[-1].replace('.', ''),
                'filename': self.model_cfg.scene_path
            }
        })

        # Load the BRDF
        brdf = hydra.utils.instantiate(OmegaConf.load(self.model_cfg["brdf"]["cfg"]))
        brdf.load_state_dict(torch.load(self.model_cfg["brdf"]["pt"], weights_only=True))
        freeze_model(brdf)

        # Initialize the per object albedo transforms
        if self.forward_cfg["use_transform"]:
            # Load the Segmentation
            segmentation = hydra.utils.instantiate(OmegaConf.load(self.model_cfg["segmentation"]["cfg"]))
            segmentation.load_state_dict(torch.load(self.model_cfg["segmentation"]["pt"], weights_only=True))
            freeze_model(segmentation)

            self.brdf = ObjTransformedBRDF(brdf_net=brdf, semantic_net=segmentation)
        else:
            self.brdf = brdf        

        # Load the CRF
        self.crf = hydra.utils.instantiate(OmegaConf.load(self.model_cfg["crf"]["cfg"]))
        self.crf.load_state_dict(torch.load(self.model_cfg["crf"]["pt"], weights_only=True))
        freeze_model(self.crf)

        # Load the Emitter
        self.emitter = hydra.utils.instantiate(OmegaConf.load(self.model_cfg["emitter"]["cfg"]))
        self.emitter.load_state_dict(torch.load(self.model_cfg["emitter"]["pt"], weights_only=True))

    def log_config(self, cfg):
        out_folder = cfg["output"]["folder_path"]
        os.makedirs(out_folder, exist_ok=True)

        # Save Emitter config
        if isinstance(self.model_cfg["emitter"]["cfg"], str):
            self.model_cfg["emitter"]["cfg"] = OmegaConf.load(self.model_cfg["emitter"]["cfg"])
        OmegaConf.save(self.model_cfg["emitter"]["cfg"], os.path.join(out_folder, f"emitter.yaml"))

        self.module_logger.info(f"Emitter config saved to {out_folder}")

    def reinit(self):
        pass

    def forward(self, x):
        raise NotImplementedError("Forward method should be implemented in the subclass.")

    def training_step(self, batch, batch_idx=None, **kwargs):
        # Get kwargs
        forward_cfg = self.forward_cfg
        forward_cfg.update(kwargs)

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

        # =================== Path Tracing ==================
        # Find the intersection of rays with the scene
        # TODO: Implement caching for these positions maps
        rays = batch['rays']
        xs,ds = rays[:,:3], rays[:,3:6]
        ds = F.normalize(ds,dim=1)
        dxdu,dydv = rays[:,6:9],rays[:,9:12]

        # Prefilter rays that do not hit the object
        # Sample within pixel
        if self.forward_cfg["jitter_within_pixel"]:
            du,dv = torch.rand(2, len(xs), 1, device=xs.device) - 0.5
            ds = F.normalize(ds + dxdu * du + dydv * dv, dim=1)

        positions, normals, _, triangle_idx, valid = ray_intersect(self.scene,xs,ds)

        if not valid.any():
            return None
        
        rgbs_gt = batch['rgbs']
        if ray_based_batch:
            rgbs_gt = rgbs_gt[valid]
            positions = positions[valid]
            xs, ds = xs[valid], ds[valid]
            dxdu, dydv = dxdu[valid], dydv[valid]

        # ================ Rendering =================
        # optimize only valid surface
        SPP = self.forward_cfg["spp"]
        spp = self.forward_cfg["spp_batch"]
        L = torch.zeros_like(xs)
        for _ in range(SPP//spp):
            L += path_tracing_single(
                self.scene, self.emitter, self.brdf, 
                xs, ds, dxdu, dydv, spp
            )
        L = L / (SPP//spp)
        
        exposure = batch['exposure'][valid]
        rgbs_hdr = L
        rgbs_ldr = self.crf(rgbs_hdr, exposure)

        # ================ Loss Calculation =================

        # Rendering loss
        loss_render = self.loss_fn(rgbs_ldr, rgbs_gt)

        # Transformation regularization loss
        if self.forward_cfg["use_transform"]:
            seen_transforms = self.brdf.get_seen_transforms()
            A, b = seen_transforms[..., :3], seen_transforms[..., 3] 
            Sigma = torch.eye(3, device=A.device)[None, ...]
            I = torch.eye(3, device=A.device)[None, ...]
            loss_albedo_transform = (((A - I) ** 2).sum(dim=(-2, -1)) + (b ** 2).sum(dim=-1) + ((A @ Sigma @ A.transpose(-1, -2) - Sigma) ** 2).sum(dim=(-2, -1))).mean()
        else:
            loss_albedo_transform = 0

        # Compose the loss
        loss = loss_render + self.loss_cfg["w_albedo_transform"] * loss_albedo_transform

        # ================ Metrics =================

        psnr = -10.0 * torch.log10(loss.clamp_min(1e-5))

        # =================== Output ==================

        output = {
            'loss': loss,
            'metric/psnr': psnr
        }

        if not ray_based_batch:
            # Add RGB
            assert B==1, "Batch size > 1 not supported for image-based training"
            rgbs_hdr = einops.rearrange(rgbs_hdr, '(h w) c -> h w c', h=H, w=W)
            denoiser = mitsuba.OptixDenoiser([H,W])
            rgbs_hdr = denoiser(rgbs_hdr.numpy(force=True)).torch().to(exposure.device)
            rgbs_hdr = einops.rearrange(rgbs_hdr, 'h w c -> (h w) c')
            rgbs_ldr = self.crf(rgbs_hdr, exposure)
            output['rgbs_render'] = einops.rearrange(rgbs_ldr, '(b h w) c -> b c h w', b=B, h=H, w=W)
        
            # Add Albedo
            mat = self.brdf(positions)
            output['albedo'] = einops.rearrange(mat["albedo"], '(b h w) c -> b c h w', b=B, h=H, w=W)
            output['roughness'] = einops.rearrange(mat["roughness"], '(b h w) c -> b c h w', b=B, h=H, w=W)
            output['metallic'] = einops.rearrange(mat["metallic"], '(b h w) c -> b c h w', b=B, h=H, w=W)

            # Add Segmentation
            if hasattr(self.brdf, 'semantic_net'):
                segmentation = self.brdf.semantic_net(positions).argmax(-1).unsqueeze(-1).long()
                hard_assignments = F.one_hot(segmentation, num_classes=self.model_cfg.num_objects).float()
                segmentation = (self.color_map[None,None,...] * hard_assignments[...,None]).sum(dim=-2).squeeze(1)
                output['segmentation'] = einops.rearrange(segmentation, '(b h w) c -> b c h w', b=B, h=H, w=W)

            # Add Emitter
            emission, _, _, emission_mask = self.emitter.eval_emitter(positions, None, triangle_idx)
            output['emission'] = einops.rearrange(emission.clamp(0,1), '(b h w) c -> b c h w', b=B, h=H, w=W)
            output['emission_mask'] = einops.rearrange(emission_mask.float(), '(b h w) -> b 1 h w', b=B, h=H, w=W)

        return output

    def configure_optimizers(self):
        params = {n:p for n, p in self.named_parameters() if p.requires_grad}
        self.module_logger.info(f"Optimizing {len(params)} parameters: {params}")

        if(self.optimizer_cfg.optimizer == 'SGD'):
            opt = torch.optim.SGD
        if(self.optimizer_cfg.optimizer == 'Adam'):
            opt = torch.optim.Adam
        
        optimizer = opt(list(params.values()), lr=self.optimizer_cfg.learning_rate, weight_decay=self.optimizer_cfg.weight_decay)    
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=self.scheduler_cfg.milestones,gamma=self.scheduler_cfg.scheduler_rate)
        return [optimizer], [scheduler]
    

