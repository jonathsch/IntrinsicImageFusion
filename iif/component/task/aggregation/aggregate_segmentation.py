import os
import einops
import hydra
import mitsuba
from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch
import torch_scatter
import torch.nn.functional as F

from iif.component.model.brdf import NGPBRDF
from iif.component.rendering.path_tracing import ray_intersect
from iif.utils.datastructure import Batch
from iif.utils.image_io import show_image
from iif.utils.logging import init_logger


def create_color_map(num_classes, seed=42):
    """Generate a color map for num_classes IDs."""
    g = torch.Generator().manual_seed(seed)  # deterministic colors
    colors = torch.randint(0, 256, (num_classes, 3), dtype=torch.uint8, generator=g) / 255
    return colors


class SegmentationAggregationModule(pl.LightningModule):
    def __init__(self, 
                 forward_cfg,
                 model_cfg, 
                 loss_cfg, 
                 optimizer_cfg,
                 scheduler_cfg,
                 *args, **kwargs):
        super(SegmentationAggregationModule, self).__init__(*args, **kwargs)
        self.module_logger = init_logger()
        self.forward_cfg = forward_cfg
        self.model_cfg = model_cfg
        self.loss_cfg = loss_cfg
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

        self.segmentation, self.scene = self.configure_model()

        self.register_buffer('color_map', create_color_map(self.model_cfg.num_objects))

    def configure_model(self):
        # Initialize the BRDF model
        # Initialize the BRDF model
        slf = torch.load(self.model_cfg["slf_path"], weights_only=True)
        self.model_cfg["material_cfg"]["voxel_min"] = slf['voxel_min'].item()
        self.model_cfg["material_cfg"]["voxel_max"] = slf['voxel_max'].item()
        segmentation_net = hydra.utils.instantiate(self.model_cfg["material_cfg"])
        if self.model_cfg.ckpt_path:
            state_dict = torch.load(self.model_cfg.ckpt_path, map_location='cpu')['state_dict']
            weight = {}
            for k,v in state_dict.items():
                if 'material.' in k:
                    weight[k.replace('material.','')]=v
            segmentation_net.load_state_dict(weight)

        # Initialize the scene - TODO: Consider delegating it to a spearate datamodule
        assert os.path.exists(self.model_cfg.scene_path), 'Mesh not found: '+ self.model_cfg.scene_path
        scene = mitsuba.load_dict({
            'type': 'scene',
            'shape_id':{
                'type': os.path.splitext(self.model_cfg.scene_path)[-1].replace('.', ''),
                'filename': self.model_cfg.scene_path
            }
        })

        return segmentation_net, scene
    
    def log_config(self, cfg):
        out_folder = cfg['output']['folder_path']
        os.makedirs(out_folder, exist_ok=True)

        # Save the segmentation config
        out_path = os.path.join(out_folder, 'segmentation.yaml')
        segmentation_config = self.model_cfg["material_cfg"]
        OmegaConf.save(segmentation_config, out_path)

        self.module_logger.info(f"Segmentation config saved to {out_path}")

    def forward(self, x):
        raise NotImplementedError("Forward method should be implemented in the subclass.")

    def training_step(self, batch, batch_idx=None):
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
        
        # find surface intersection
        positions, normals, _, triangle_idx, valid = ray_intersect(self.scene,xs,ds)

        if not valid.any():
            return None

        # ================ Segmentation Prediction =================
        # Forward the positions through the model
        segmentation = self.segmentation(positions)
        segmentation_ref = batch['segmentation'].long()
       
        # ================ Loss Calculation =================
        if self.loss_cfg["use_mask"]:
            mask = (segmentation_ref.squeeze(-1) > 0).float()
            loss_segmentation = (F.cross_entropy(segmentation, segmentation_ref.squeeze(-1), reduction='none') * mask).mean()
        else:
            loss_segmentation = F.cross_entropy(segmentation, segmentation_ref.squeeze(-1))

        loss = loss_segmentation

        output = {
            'loss': loss,
            'loss_segmentation': loss_segmentation,
        }
        if not ray_based_batch:
            output["segmentation_gt"] = {
                "hdr": batch["segmentation"].long(),
                "ldr": self.get_ldr_segmentation(batch["segmentation"].long())
            }

            segmentation = torch.argmax(segmentation,dim=-1).unsqueeze(-1)
            output['segmentation'] = {
                'hdr': torch.round(segmentation.long()),
                'ldr': self.get_ldr_segmentation(torch.round(segmentation.long()))
            }

            output["segmentation_gt"] = Batch(output["segmentation_gt"]).map(lambda x: einops.rearrange(x, '(b h w) c -> b c h w', b=B, h=H, w=W)).to_dict()
            output['segmentation'] = Batch(output['segmentation']).map(lambda x: einops.rearrange(x, '(b h w) c -> b c h w', b=B, h=H, w=W)).to_dict()
        return output
    
    def get_ldr_segmentation(self, segmentation_hdr):
        segmentation_hdr = segmentation_hdr.clamp(0,self.model_cfg.num_objects-1)
        hard_assignments = F.one_hot(segmentation_hdr, num_classes=self.model_cfg.num_objects).float()
        return (self.color_map[None,None,...] * hard_assignments[...,None]).sum(dim=-2).squeeze(1)

    def configure_optimizers(self):
        if(self.optimizer_cfg.optimizer == 'SGD'):
            opt = torch.optim.SGD
        if(self.optimizer_cfg.optimizer == 'Adam'):
            opt = torch.optim.Adam
        
        optimizer = opt(self.parameters(), lr=self.optimizer_cfg.learning_rate, weight_decay=self.optimizer_cfg.weight_decay)    
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=self.scheduler_cfg.milestones,gamma=self.scheduler_cfg.scheduler_rate)
        return [optimizer], [scheduler]
    

