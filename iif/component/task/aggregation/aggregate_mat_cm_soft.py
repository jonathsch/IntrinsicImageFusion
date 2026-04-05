import os
import einops
import hydra
import mitsuba
from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch
import torch_scatter
import torch.nn.functional as F

from iif.component.model.brdf import NGPBRDF, NGPAssignment, ProbabilisticNGPBRDF
from iif.component.rendering.path_tracing import ray_intersect
from iif.utils.datastructure import Batch
from iif.utils.image_io import show_image
from iif.utils.logging import init_logger
from iif.utils.model import get_config


class SoftCMMaterialAggregationModule(pl.LightningModule):
    def __init__(self, 
                 forward_cfg,
                 model_cfg, 
                 loss_cfg, 
                 optimizer_cfg,
                 scheduler_cfg,
                 *args, **kwargs):
        self.module_logger = init_logger()
        super(SoftCMMaterialAggregationModule, self).__init__(*args, **kwargs)
        self.forward_cfg = Batch(self.default_forward_cfg())
        self.forward_cfg.update(forward_cfg)

        self.model_cfg = Batch(model_cfg)
        self.loss_cfg = Batch(self.default_loss_cfg())
        self.loss_cfg.update(loss_cfg)
                             
        self.optimizer_cfg = Batch(optimizer_cfg)
        self.scheduler_cfg = Batch(scheduler_cfg)

        self.configure_model()

        self.loss_fn = F.l1_loss
    
    def default_forward_cfg(self):
        return {
            "scale_to_gt": False,
            "use_transform": False,
            "jitter_within_pixel": True,
            "use_st": False,
            "use_mixture": True,
            "use_error_dist": False
        }
    
    def default_loss_cfg(self):
        return {
            "temperature_logit": 1.0,
            "temperature_error": 1.0,
            "w_albedo_transform": 0.1,
        }

    def configure_model(self):
        # Initialize the scene - TODO: Consider delegating it to a spearate datamodule
        assert os.path.exists(self.model_cfg["scene_path"]), 'Mesh not found: '+ self.model_cfg.scene_path
        self.scene = mitsuba.load_dict({
            'type': 'scene',
            'shape_id':{
                'type': os.path.splitext(self.model_cfg["scene_path"])[-1].replace('.', ''),
                'filename': self.model_cfg["scene_path"]
            }
        })

        # Load the SLF
        slf = hydra.utils.instantiate(self.model_cfg["slf"])

        # Update the BRDF config with the SLF
        self.model_cfg["brdf"] = self.update_brdf_cfg(self.model_cfg["brdf"], slf)

        # Initialize the BRDF model
        self.brdf_cfg = get_config(self.model_cfg["brdf"])
        self.brdf = hydra.utils.instantiate(self.model_cfg["brdf"])

        # Initialize the soft assignment logits
        self.assignment_logits = torch.nn.Parameter(torch.zeros((self.model_cfg["num_segments"], self.model_cfg["num_predictions_per_image"], 3))) 

        # Initialize the BRDF transforms
        self.brdf_prediction_transform = hydra.utils.instantiate(self.model_cfg["brdf_prediction_transform"])
            
    def update_brdf_cfg(self, brdf_cfg, slf):
        overrides = {
            "_voxel_min_": slf.voxel_min.item(),
            "_voxel_max_": slf.voxel_max.item(),
        }
        for key, value in brdf_cfg.items():
            if value in overrides:
                brdf_cfg[key] = overrides[value]
            elif isinstance(value, dict):
                brdf_cfg[key] = self.update_brdf_cfg(value, slf)
        return brdf_cfg

    def log_config(self, cfg):
        out_folder = cfg["output"]["folder_path"]
        os.makedirs(out_folder, exist_ok=True)

        # Save BRDF config
        OmegaConf.save(self.brdf_cfg.to_dict(), os.path.join(out_folder, f"brdf.yaml"))

        self.module_logger.info(f"Configs saved to {out_folder}")

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

        # =================== Ray-Scene Intersection ==================
        # Find the intersection of rays with the scene
        rays = batch['rays']
        xs,ds = rays[:,:3], rays[:,3:6]
        ds = F.normalize(ds,dim=1)
        dxdu,dydv = rays[:,6:9],rays[:,9:12]

        # Sample within pixel
        if forward_cfg["jitter_within_pixel"]:
            du,dv = torch.rand(2, len(xs), 1, device=xs.device) - 0.5
            ds = F.normalize(ds + dxdu * du + dydv * dv, dim=1)

        positions, normals, _, triangle_idx, valid = ray_intersect(self.scene,xs,ds)

        if not valid.any():
            return None

        # ================ Material Prediction =================
        # Forward the positions through the model
        mat = self.brdf(positions)
        albedo, albedo_std = mat['albedo'], mat['albedo_std']
        roughness, roughness_std = mat['roughness'], mat['roughness_std']
        metallic, metallic_std = mat['metallic'], mat['metallic_std']
        mat_pred = torch.cat([albedo, roughness, metallic], dim=-1)
        mat_pred_std = torch.cat([albedo_std, roughness_std, metallic_std], dim=-1)

        # Transform the image predictions
        mat_ref = Batch(albedo=batch['albedo_ref'],
                        roughness=batch['roughness_ref'],
                        metallic=batch['metallic_ref'])
        segment_idx = batch["per_image_segmentation"].squeeze(-1).long()
        mat_ref = self.brdf_prediction_transform(segment_idx, mat_ref)

        albedo_ref = mat_ref['albedo']
        roughness_ref = mat_ref['roughness']
        metallic_ref = mat_ref['metallic']

        mat_ref = torch.cat([albedo_ref, roughness_ref, metallic_ref], dim=-1)

        # ================ Loss Calculation =================

        # Collect all losses
        loss_info = Batch()

        # 1. Per-prediction errors
        material_residuals = self.loss_fn(mat_pred.unsqueeze(1).expand_as(mat_ref), mat_ref, reduction='none')
        material_errors = torch.cat([material_residuals[..., 0:3].mean(dim=-1, keepdim=True), material_residuals[..., 3:4], material_residuals[..., 4:5]], dim=-1)

        # 2. Soft target assignment
        assignment_logits = self.assignment_logits[segment_idx]
        assignment_soft = torch.softmax(assignment_logits / self.loss_cfg["temperature_logit"], dim=1)
        assignment_soft_expanded = torch.cat([assignment_soft[..., 0:1], assignment_soft[..., 0:1], assignment_soft[..., 0:1], assignment_soft[..., 1:2], assignment_soft[..., 2:3]], dim=-1)
        # assignment_hard = assignment_logits.argmax(dim=-2)

        # 3. Data loss
        mat_mixed_ref = (assignment_soft_expanded * mat_ref).sum(dim=1)
        mat_mixed_ref_std = torch.abs(mat_ref - mat_mixed_ref.unsqueeze(1)).median(dim=1).values

        albedo_ref_dist = torch.distributions.Laplace(mat_mixed_ref, mat_mixed_ref_std.clamp(1e-2))
        albedo_model_dist = torch.distributions.Laplace(mat_pred, mat_pred_std.clamp(1e-2))
        loss_info["brdf"] = torch.distributions.kl.kl_divergence(albedo_ref_dist, albedo_model_dist).mean()

        # inlier_weight = torch.exp( - albedo_residuals / albedo_mixed_ref_std.unsqueeze(1).clamp(1e-2))
        # loss_albedo = (inlier_weight * s_soft[..., None] * albedo_residuals).sum(dim=1).mean()

        # 4. Label loss
        q = torch.softmax(-material_errors / self.loss_cfg["temperature_error"], dim=1).detach()
        logp = torch.log_softmax(assignment_logits / self.loss_cfg["temperature_logit"], dim=1)
        loss_info["label"] = - (q * logp).sum(dim=1).mean()

        # 6. Transform regularization loss
        # Add regularizations
        if "brdf_prediction_transform" in self.loss_cfg["regularizations"]:
            loss_info["brdf_prediction_transform"] = Batch(self.brdf_prediction_transform.get_regularization_loss())

        # Compose the loss
        loss_info = loss_info.flatten(separator="_")
        weight = Batch(dict(self.loss_cfg["weight"]), default=lambda: 1.0)
        loss = sum(list((loss_info * weight).values()))

        # ================ Metrics =================
        # Entropy
        albedo_assignment_entropy = -(assignment_soft * torch.log(assignment_soft + 1e-8)).sum(dim=-1).mean()

        output = {
            'loss': loss,
            "debug_logits_entropy": albedo_assignment_entropy,
        }

        loss_info = loss_info.map_keys(lambda x: f'loss/{x}')
        output.update(loss_info.to_dict())

        if not ray_based_batch:
            output['material'] = mat
            output["material"]["reference"] = Batch(albedo_ref=albedo_ref).map(lambda x: einops.rearrange(x, '(b h w) p c -> (b p h w) c', h=H, w=W)).to_dict()
            output["material"]["reference_mixture_mean"] = Batch(albedo_ref=mat_mixed_ref[:, :3])
            output["material"]["reference_mixture_std"] = Batch(albedo_ref=mat_mixed_ref_std[:, :3]) * 5
            
            output["material"]["assignment"] = Batch(assignment=torch.softmax(assignment_logits / self.loss_cfg["temperature_logit"], dim=1)).map(lambda x: einops.rearrange(x, '(b h w) p c -> (b p h w) c', h=H, w=W)).to_dict()
            if forward_cfg["scale_to_gt"]:
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
                
                def fit_and_transform(pred, target):
                    pred = torch.cat([pred, torch.ones_like(pred[..., :1])], dim=-1)
                    
                    obj_transforms = []
                    obj_segmentation = batch["segmentation"].squeeze(-1).long()
                    segment_segmentation = obj_segmentation.clone()
                    obj_ids = obj_segmentation.unique()   
                    for obj_idx, obj_id in enumerate(obj_ids):
                        mask = obj_segmentation == obj_id
                        segment_segmentation[mask] = obj_idx
                        obj_transforms.append(affine_transform_torch(pred, target, mask))
                    obj_transforms = torch.stack(obj_transforms, dim=0)  # (num_obj, 3, 4)
                    obj_transforms = torch.gather(obj_transforms[None, ...].expand(segment_segmentation.shape[0], -1, -1, -1), 
                                                1, 
                                                segment_segmentation[:, None, None, None].expand(-1, 1, *obj_transforms.shape[1:])).squeeze(1)  # (N, 3, 4)
 
                    return einops.einsum(pred, obj_transforms, "B D, B C D -> B C").clamp(0,1).nan_to_num(0)
                
                # Transform the albedo
                albedo_gt = batch["albedo"]
                albedo_pred = output['material']['albedo']
                output['material']['albedo_transformed'] = fit_and_transform(albedo_pred, albedo_gt)
                
                # Transform the roughness
                roughness_gt = batch["roughness"]
                roughness_pred = output['material']['roughness']
                output['material']['roughness_transformed'] = fit_and_transform(roughness_pred, roughness_gt)

                # Transform the metallic
                metallic_gt = batch["metallic"]
                metallic_pred = output['material']['metallic']
                output['material']['metallic_transformed'] = fit_and_transform(metallic_pred, metallic_gt)

            output['material'] = Batch(output['material']).map(lambda x: einops.rearrange(x, '(b h w) c -> b c h w', h=H, w=W)).to_dict()
        return output

    def configure_optimizers(self):
        if(self.optimizer_cfg["optimizer"] == 'SGD'):
            opt = torch.optim.SGD
        if(self.optimizer_cfg["optimizer"] == 'Adam'):
            opt = torch.optim.Adam
        
        optimizer = opt(self.parameters(), lr=self.optimizer_cfg["learning_rate"], weight_decay=self.optimizer_cfg["weight_decay"])    
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=self.scheduler_cfg["milestones"],gamma=self.scheduler_cfg["scheduler_rate"])
        return [optimizer], [scheduler]
    

