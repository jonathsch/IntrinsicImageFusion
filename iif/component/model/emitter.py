# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import einops
import torch
import torch.nn as nn
import torch.nn.functional as NF
import numpy as np
import math
import argparse
import tinycudann as tcnn

from iif.utils.batching import batched_average
from iif.utils.datastructure import Batch
from iif.utils.logging import init_logger
from .slf import VoxelSLF

class AreaEmitter(nn.Module):
    """ triangle mesh emitters """
    def __init__(self,emitter_path):
        """ emitter_path file 
        is_emitter: B indicator of whether a triangle is emitter
        emitter_vertices: Kx3x3 triangle vertices of emitters
        emitter_area: K surface areas of emitters
        emitter_radiance: Bx3x3 emitter radiance
        """
        super(AreaEmitter,self).__init__()
        
        weight = torch.load(emitter_path,map_location='cpu')
        
        is_emitter = weight['is_emitter']
        emitter_vertices = weight['emitter_vertices']
        emitter_area = weight['emitter_area']
        emitter_radiance = weight['emitter_radiance']

        self.register_buffer('is_emitter',is_emitter)
        self.register_buffer('emitter_vertices',emitter_vertices)
        self.register_buffer('emitter_area',emitter_area)
        self.register_buffer('radiance',emitter_radiance)
        
        # emitter idx mapping, -1 indicates not an emitter
        emitter_idx = torch.full((len(is_emitter),),-1,device=is_emitter.device,dtype=torch.long)
        emitter_idx[is_emitter] = torch.arange(is_emitter.sum(),device=is_emitter.device)
        self.register_buffer('emitter_idx',emitter_idx)
        
        # emitter idx to triangle idx
        triangle_idx = torch.arange(len(is_emitter))[is_emitter]
        self.register_buffer('triangle_idx',triangle_idx)
        
        # sample emitters uniformly
        emitter_pdf = NF.normalize(torch.ones_like(emitter_area),dim=-1,p=1)
        emitter_cdf = emitter_pdf.cumsum(-1).contiguous()
        self.register_buffer('emitter_pdf',emitter_pdf)
        self.register_buffer('emitter_cdf',emitter_cdf)
    
    def forward(self,triangle_idx):
        """ get emitter radiance
        triangle_idx: B triangle indices
        """
        vis = triangle_idx != -1 # whether a valid triangle

        is_area = self.is_emitter[triangle_idx]&vis
        Le = torch.zeros(position.shape[0],3,device=position.device)
        if is_area.any():
            e_idx = self.emitter_idx[triangle_idx[is_area]]
            Le[is_area] = self.radiance[e_idx]
        
        # assume zero background lighting
        Le = Le*vis[...,None]
        return Le
    
    def eval_emitter(self, position,light_dir,triangle_idx,*args):
        """ evaluate surface emission and pdf
        Args:
            position: Bx3 intersection location
            light_dir: Bx3 emission direction
            triangle_idx: B intersected triangle id
        Return:
            Le: Bx3 radiance
            emit_pdf: Bx1 emitter pdf
            valid_next: B valid surface
        """
        # whether valid intersection
        vis = triangle_idx != -1

        # get area light
        is_area = self.is_emitter[triangle_idx]&vis

        Le = torch.zeros(position.shape[0],3,device=position.device)
        emit_pdf = torch.zeros(position.shape[0],device=position.device)
        if is_area.any():
            e_idx = self.emitter_idx[triangle_idx[is_area]]
            emit_pdf[is_area] = self.emitter_pdf[e_idx]/self.emitter_area[e_idx].clamp_min(1e-12)
            Le[is_area] = self.radiance[e_idx]

        # assume zero background lighting
        Le = Le*vis[...,None]

        # next: not area light or background
        valid_next = (~is_area)&vis
        return Le,emit_pdf.unsqueeze(-1),valid_next
    
    def sample_emitter(self,sample1,sample2,position):
        """ importance sampling emitters
        Args:
            sample1: B uniform samples
            sample2: Bx2 uniform samples
            position: Bx3 surfae location
        Return:
            wi: Bx3 sampled direction
            pdf: Bx1 the sampling pdf (in area space)
            triangle_idx: B the sampled triangle id
        """
        # pick an emitter
        emitter_idx = torch.searchsorted(self.emitter_cdf,sample1.clamp_min(1e-12))
        pdf0 = self.emitter_pdf[emitter_idx]

        # unifromly sample points on triangles
        xi1 = sample2[...,0].sqrt()
        u = (1-xi1).unsqueeze(-1)
        v = (xi1*sample2[...,1]).unsqueeze(-1)
        w = 1-u-v

        # emitter area
        A1 = self.emitter_area[emitter_idx]
        # sampled location on triangle
        p1 = self.emitter_vertices[emitter_idx]
        p1 = p1[:,0]*u + p1[:,1]*v + p1[:,2]*w
        wi = NF.normalize(p1-position,dim=-1)
        triangle_idx = self.triangle_idx[emitter_idx]
        
        # pdf in area space
        pdf = pdf0/A1.clamp_min(1e-12)
        return wi,pdf.unsqueeze(-1),triangle_idx
    
def test():
    emitter_gt_path = 'outputs/0703_kitchen_hdr/bake/emitter.pth'
    emitter_gt = torch.load(emitter_gt_path, map_location='cpu')
    radiance_gt = emitter_gt['emitter_radiance'].numpy()
    area_gt = emitter_gt['emitter_area'].numpy()
    emitter_learn_path = 'outputs/0721_kitchen_init_albedo_1/bake/emitter.pth'
    emitter_learn = torch.load(emitter_learn_path, map_location='cpu')
    radiance_learn = emitter_learn['emitter_radiance'].numpy()
    area_learn = emitter_learn['emitter_area'].numpy()
    # print(radiance_gt.shape)

    radiance_gt = radiance_gt # * area_gt[:, None] / area_gt.sum()
    radiance_learn = radiance_learn #* area_learn[:, None] / area_learn.sum()
    print('[GT from HDR]      min: {:.5f}, max: {:.5f}, mean: {:.5f}'.format(radiance_gt.min(), radiance_gt.max(), radiance_gt.mean()))
    print('[learned from LDR] min: {:.5f}, max: {:.5f}, mean: {:.5f}'.format(radiance_learn.min(), radiance_learn.max(), radiance_learn.mean()))
    print('Ratio of Mean: {:.5f}'.format(radiance_learn.mean()/radiance_gt.mean()) )
    print('Mean of Ratio: {:.5f}'.format(np.mean(radiance_learn / radiance_gt.clip(min=1e-4))))




class NGPEmitter(nn.Module):
    """ Hash Grid based brdf paramterization """
    def __init__(self,
                 voxel_min=0,
                 voxel_max=1):
        """ 
        voxel_min,voxel_max: scene bounding box
        """
        super(NGPEmitter,self).__init__()

        hash_encoding={
                "otype": "HashGrid",
                "n_levels": 32,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.3
         }
        
        hash_network={
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": 64,
            "n_hidden_layers": 2
        }

        self.mlp =  tcnn.NetworkWithInputEncoding(
            n_input_dims=3, 
            n_output_dims=3, 
            encoding_config=hash_encoding, 
            network_config=hash_network)
        
        self.register_buffer('voxel_min', torch.tensor(voxel_min))
        self.register_buffer('voxel_max', torch.tensor(voxel_max))

        
    def forward(self,position, **kwargs):
        """ query brdf parameters at given location
        Args:
            position: Bx3 queried location
        Return:
            Bx3 base color
            Bx1 roughness in [0.02,1]
            Bx1 metallic
        """
        # map to [0,1]
        position = (position-self.voxel_min)/(self.voxel_max-self.voxel_min)
        
        emission = self.mlp(position*2-1).exp()
        return emission
    
    def eval_emitter(self, 
                     position, 
                     light_dir=None, 
                     triangle_idx=None,
                     roughness=None, 
                     trace_roughness=0.6):
        return self.forward(position), None, None, None


class SphereEmitter(nn.Module):
    def __init__(self, center=torch.zeros(3), radius=1.0, radiance=torch.ones(3)):
        super(SphereEmitter, self).__init__()
        
        self.center = nn.Parameter(center)
        self.radius = nn.Parameter(torch.tensor(radius))
        self.radiance = nn.Parameter(radiance)


class SLFEmitter(nn.Module):
    """ triangle emitters with diffuse radiance cache """
    def __init__(self, grid_size=1, n_triangles=0, n_emitters=0, activation="relu", caching=True):
        """ 
        emitter_path: emitter parameter file
        slf_path: surface light field paramter file
        """
        super(SLFEmitter,self).__init__()
        self.module_logger = init_logger()
        
        # load surface light field
        self.slf = VoxelSLF(grid_size=grid_size)
        
        # Define placeholder buffers
        self.register_buffer('is_emitter', torch.zeros(n_triangles, dtype=torch.bool))
        self.register_buffer('original_is_emitter', torch.zeros(n_triangles, dtype=torch.bool))
        self.register_buffer('emitter_vertices', torch.zeros(n_emitters, 3, 3))
        self.register_buffer('emitter_area', torch.zeros(n_emitters, 3))
        self.register_buffer('emitter_normal', torch.zeros(n_emitters, 3))

        # Define trainable parameters
        self.radiance = nn.Parameter(torch.zeros(n_emitters, 3))
        self.register_buffer('valid_emitter', torch.zeros(n_emitters, dtype=torch.bool))
        
        # emitter idx mapping, -1 indicates not an emitter
        emitter_idx = torch.full((n_triangles,), -1, dtype=torch.long)
        self.register_buffer('emitter_idx',emitter_idx)
        
        # emitter idx to triangle idx
        self.register_buffer('triangle_idx', torch.empty(0, 3))
        
        # randomly select a emitter
        self.register_buffer('emitter_pdf', torch.zeros(n_triangles, 3))
        self.register_buffer('emitter_cdf', torch.zeros(n_triangles, 3))

        # Cache for regularization
        self.caching = caching
        if self.caching:
            self.regularization_info = Batch(default=list)

        # Define the ativation
        self.activation = activation

    def load_state_dict(self, state_dict, *args, **kwargs):
        self.is_emitter = state_dict['is_emitter']
        self.emitter_vertices = state_dict['emitter_vertices']
        self.emitter_area = state_dict['emitter_area']
        self.emitter_normal = state_dict['emitter_normal']
        # self.radiance.data = torch.log(state_dict['radiance'].clamp(1e-12) * 100)
        self.radiance.data = state_dict['radiance']
        self.valid_emitter = state_dict['valid_emitter']

        self.emitter_idx = state_dict['emitter_idx']
        self.triangle_idx = state_dict['triangle_idx']

        self.emitter_pdf = state_dict['emitter_pdf']
        self.emitter_cdf = state_dict['emitter_cdf']

        # TODO: Remove this backward compatibility
        if 'original_is_emitter' in state_dict:
            self.original_is_emitter = state_dict['original_is_emitter']
        else:
            self.original_is_emitter = self.is_emitter.clone()

        self.slf.load_state_dict({k[4:]: v for k,v in state_dict.items() if k.startswith('slf.')}, *args, **kwargs)

    # def save(self, out_path):
    #     # Collate the pruned parameters
    #     emitter_keep_mask = self.is_emitter & self.original_is_emitter

    #     # Prune the parameters
    #     self.radiance = 

    def initialize(self, 
                   is_emitter,
                   emitter_vertices,
                   emitter_area,
                   emitter_normal,
                   emitter_radiance,
                   slf):
        self.is_emitter = is_emitter
        self.original_is_emitter = is_emitter.clone()
        self.emitter_vertices = emitter_vertices
        self.emitter_area = emitter_area
        self.emitter_normal = emitter_normal
        self.radiance.data = emitter_radiance
        self.valid_emitter = torch.ones_like(self.radiance.data[:, 0], dtype=torch.bool)
        self.slf = slf

        # emitter idx mapping, -1 indicates not an emitter
        emitter_idx = torch.full((len(is_emitter),),-1,device=is_emitter.device,dtype=torch.long)
        emitter_idx[is_emitter] = torch.arange(is_emitter.sum(),device=is_emitter.device)
        self.emitter_idx = emitter_idx
        
        # emitter idx to triangle idx
        triangle_idx = torch.arange(len(is_emitter))[is_emitter]
        self.triangle_idx = triangle_idx
        
        # randomly select a emitter
        emitter_pdf = NF.normalize(torch.ones_like(emitter_area),dim=-1,p=1)
        emitter_cdf = emitter_pdf.cumsum(-1).contiguous()
        self.emitter_pdf = emitter_pdf
        self.emitter_cdf = emitter_cdf
    
    def forward(self, position):
        """ surface light field from queried location """
        Le = self.slf(position)['rgb']
        return Le
    
    def eval_emitter(self, 
                     position, 
                     light_dir, 
                     triangle_idx,
                     roughness=None, 
                     trace_roughness=0.6):
        """ evaluate surface emission and pdf return radiance cache if diffuse
        Args:
            position: Bx3 intersection location
            light_dir: Bx3 emission direction
            triangle_idx: B intersected triangle id
            roughness: Bx1 surface roughness if not None
        Return:
            Le: Bx3 radiance
            emit_pdf: Bx1 emitter pdf
            valid_next: B valid surface
        """
        # whether valid intersection
        vis = triangle_idx != -1
        
        Le = torch.zeros(position.shape[0],3,device=position.device)
        emit_pdf = torch.zeros(position.shape[0],device=position.device)
        
        # get area light
        is_area = self.is_emitter[triangle_idx] & vis
        if is_area.any():
            # self.module_logger.debug(f"is_area {is_area}, triangle_idx {triangle_idx}, emitter_idx {self.emitter_idx}")
            # self.module_logger.debug(f"Evaluating emitters at {triangle_idx[is_area]}")
            e_idx = self.emitter_idx[triangle_idx[is_area]]
            # assert not (e_idx == -1).any(), f"Invalid emitter index found: e_idx {e_idx}, triangle_idx {triangle_idx[is_area]}, self.emitter_idx {self.emitter_idx}."
            emit_pdf[is_area] = self.emitter_pdf[e_idx]/self.emitter_area[e_idx].clamp_min(1e-12)

            # radiance = self.radiance[e_idx].clamp(0)
            radiance = self.get_radiance()[e_idx]
            Le[is_area] = radiance
            # Save the regularization info
            if self.caching and torch.is_grad_enabled():
                self.regularization_info["radiance"].append(radiance)
        
        # assume zero background lighting
        Le = Le*vis[...,None]
        valid_next = (~is_area)&vis

        # check diffuse radiance cache
        if isinstance(roughness, (int, float)) and roughness == -1:
            # for regularization purpose, always query the cache
            is_diffuse = (~is_area) & vis
            if is_diffuse.any():
                diffuse_slf = self.slf(position[is_diffuse])['rgb']
                L_diffuse = torch.zeros_like(Le)
                L_diffuse[is_diffuse] = diffuse_slf
                Le = Le + L_diffuse
                is_diffuse[is_diffuse.clone()] = diffuse_slf.sum(-1) > 0 # diffuse radiance need to > 0
                valid_next &= (~is_diffuse) # terminate path 
        elif roughness is not None:
            # query the radiance cache and terminate for diffuse and non emissive surface 
            is_diffuse = (~is_area) & vis & (roughness.squeeze(-1)>trace_roughness)
            if is_diffuse.any():
                diffuse_slf = self.slf(position[is_diffuse])['rgb']
                L_diffuse = torch.zeros_like(Le)
                L_diffuse[is_diffuse] = diffuse_slf
                Le = Le + L_diffuse
                is_diffuse[is_diffuse.clone()] = diffuse_slf.sum(-1) > 0 # diffuse radiance need to > 0
                valid_next &= (~is_diffuse) # terminate path 

        return Le, emit_pdf.unsqueeze(-1), valid_next, is_area
    

    def sample_emitter(self,sample1,sample2,position):
        """ importance sampling emitters
        Args:
            sample1: B uniform samples
            sample2: Bx2 uniform samples
            position: Bx3 surfae location
        Return:
            wi: Bx3 sampled direction
            pdf: Bx1 the sampling pdf (in area space)
            triangle_idx: B the sampled triangle id
        """
        # pick an emitter
        emitter_idx = torch.searchsorted(self.emitter_cdf,sample1)
        emitter_idx.clamp_(0, self.emitter_cdf.shape[0]-1)
        pdf0 = self.emitter_pdf[emitter_idx]

        # unifromly sample points on triangles
        xi1 = sample2[...,0].sqrt()
        u = (1-xi1).unsqueeze(-1)
        v = (xi1*sample2[...,1]).unsqueeze(-1)
        w = 1-u-v

        # emitter area
        A1 = self.emitter_area[emitter_idx]
        # sampled location on triangle
        p1 = self.emitter_vertices[emitter_idx]
        p1 = p1[:,0]*u + p1[:,1]*v + p1[:,2]*w
        wi = NF.normalize(p1-position,dim=-1)
        triangle_idx = self.triangle_idx[emitter_idx]
        
        # pdf in area space
        pdf = pdf0/A1.clamp_min(1e-12)
        return wi,pdf.unsqueeze(-1),triangle_idx

    def log_details(self, positions, directions, triangle_idx, b, h, w, spp, spp_batch, emission_gt=None):
        output = Batch()

        emission, _, _, emission_mask = batched_average(self.eval_emitter, 
                                                    Batch(position=einops.rearrange(positions, "(b spp) ... -> b spp ...", spp=spp), 
                                                            light_dir=einops.rearrange(directions, "(b spp) ... -> b spp ...", spp=spp), 
                                                            triangle_idx=einops.rearrange(triangle_idx, "(b spp) ... -> b spp ...", spp=spp)),
                                                    spp, spp_batch)
        emission_mask = emission_mask > 0.
        # emission, _, _, emission_mask = self.eval_emitter(positions, directions, triangle_idx)
        # emission = einops.rearrange(emission, "(b spp) c -> b spp c", spp=spp).mean(1)
        # emission_mask = einops.rearrange(emission_mask, "(b spp) -> b spp", spp=spp).any(dim=1)
        output['radiance'] = einops.rearrange(emission.clamp(0,1), '(b h w) c -> b c h w', b=b, h=h, w=w)
        output['mask'] = einops.rearrange(emission_mask.float(), '(b h w) -> b 1 h w', b=b, h=h, w=w)
        output['num_emitter_triangles'] = self.is_emitter.sum().float().unsqueeze(0)

        if emission_gt is not None:
            output["radiance_error"] = (einops.rearrange(emission - emission_gt, '(b h w) c -> b c h w', b=b, h=h, w=w) + 0.5).clamp(0,1)

        return output
    
    def get_regularization_loss(self):
        assert self.caching, "Regularization caching is disabled."

        # Retrieve from the cache
        emitter_radiance = torch.cat(self.regularization_info["radiance"], dim=0)

        # Clear the cache
        self.regularization_info = Batch(default=list)

        # Calculate regularization
        monochrome_regularization = emitter_radiance.std(dim=-1).mean()

        return Batch(
            monochrome=monochrome_regularization,
            radiance=emitter_radiance.mean()
        )

    def get_radiance(self):
        # return self.radiance.exp()
        if self.activation == "relu":
            return self.radiance.clamp(0.) * self.valid_emitter.unsqueeze(-1)
        elif self.activation == "exp":
            # Activation combining exp and relu to avoid vanishing gradient
            return (self.radiance.clamp(0.).exp() - 1.) * self.valid_emitter.unsqueeze(-1)
    
    @torch.no_grad()
    def prune_emitters(self, absolut_threshold=1., relative_threshold=0.1, percentile_threshold=0.1):
        """ Prune emitters based on their radiance values.
            Args:
                absolut_threshold: Absolute threshold for radiance values.
                relative_threshold: Relative threshold based on the mean radiance.
                percentile_threshold: Percentile threshold to prune the lowest emitters.
        """
        # Get current radiance values
        emitter_radiance = self.get_radiance().sum(dim=-1)

        # Determine thresholds
        thresholds = []
        if absolut_threshold is not None:
            thresholds.append(absolut_threshold)
        if relative_threshold is not None:
            thresholds.append(relative_threshold * emitter_radiance.max().item())
        if percentile_threshold is not None:
            thresholds.append(torch.quantile(emitter_radiance[emitter_radiance > 0], percentile_threshold).item())
        
        if len(thresholds) == 0:
            self.module_logger.info("No thresholds provided for pruning. Skipping pruning step.")
            return
        
        final_threshold = max(thresholds)
        # Identify emitters to keep
        emitter_keep_mask = (emitter_radiance > final_threshold)
        num_pruned = (~emitter_keep_mask).sum().item()
        remaining_emitters = emitter_keep_mask.sum().item()

        self.module_logger.info(f"Pruning {num_pruned} emitters from {emitter_radiance} (Threshold: {final_threshold:.4f})")
        self.module_logger.info(f"Remaining emitters: {remaining_emitters}")

        if remaining_emitters == 0:
            self.module_logger.warning("All emitters have been pruned. At least one emitter must remain.")
            return

        # Prune the parameters
        triangle_keep_mask = self.original_is_emitter.clone()
        triangle_keep_mask[self.original_is_emitter] = emitter_keep_mask
        self.is_emitter[~triangle_keep_mask] = False
        # self.emitter_vertices = self.emitter_vertices[emitter_keep_mask]
        # self.emitter_area = self.emitter_area[emitter_keep_mask]
        # self.emitter_normal = self.emitter_normal[emitter_keep_mask]

        # self.radiance.data[~emitter_keep_mask] = 0.  # Don't completely override here, because this is an optimizable parameter
        # self.radiance.data[~emitter_keep_mask] = -100.  # Don't completely override here, because this is an optimizable parameter
        self.valid_emitter = emitter_keep_mask

        # Update emitter_idx and triangle_idx
        # emitter_idx = torch.full((len(self.is_emitter),), -1, device=self.is_emitter.device, dtype=torch.long)
        # emitter_idx[self.is_emitter] = torch.arange(self.is_emitter.sum(), device=self.is_emitter.device)
        # self.emitter_idx = emitter_idx
        # triangle_idx = torch.arange(len(self.is_emitter), device=self.is_emitter.device)[self.is_emitter]
        # self.triangle_idx = triangle_idx

        self.emitter_idx[~triangle_keep_mask] = -1

        # # Update emitter_pdf and emitter_cdf
        # emitter_pdf = torch.ones_like(self.emitter_area)
        # self.emitter_pdf[~emitter_keep_mask] = 0.
        # emitter_pdf = NF.normalize(torch.ones_like(self.emitter_area), dim=-1, p=1)
        # emitter_cdf = emitter_pdf.cumsum(-1).contiguous()
        # self.emitter_pdf = emitter_pdf
        # self.emitter_cdf = emitter_cdf + (1 + 1e-12 - emitter_cdf[-1])  # Ensure the last value is slightly larger than 1.0

        self.emitter_pdf[~emitter_keep_mask] = 0.
        self.emitter_pdf = NF.normalize(self.emitter_pdf, dim=-1, p=1)
        self.emitter_cdf = self.emitter_pdf.cumsum(-1).contiguous()
        


class SLFImportanceEmitter(SLFEmitter):
    """ triangle emitters with diffuse radiance cache """
    def __init__(self, grid_size=1, n_triangles=0, n_emitters=0, activation="relu", caching=True):
        """ 
        emitter_path: emitter parameter file
        slf_path: surface light field paramter file
        """
        super(SLFEmitter,self).__init__()
        self.module_logger = init_logger()
        
        # load surface light field
        self.slf = VoxelSLF(grid_size=grid_size)
        
        # Define placeholder buffers
        self.register_buffer('is_emitter', torch.zeros(n_triangles, dtype=torch.bool))
        self.register_buffer('original_is_emitter', torch.zeros(n_triangles, dtype=torch.bool))
        self.register_buffer('emitter_vertices', torch.zeros(n_emitters, 3, 3))
        self.register_buffer('emitter_area', torch.zeros(n_emitters, 3))
        self.register_buffer('emitter_normal', torch.zeros(n_emitters, 3))

        # Define trainable parameters
        self.radiance = nn.Parameter(torch.zeros(n_emitters, 3))
        self.register_buffer('valid_emitter', torch.zeros(n_emitters, dtype=torch.bool))
        
        # emitter idx mapping, -1 indicates not an emitter
        emitter_idx = torch.full((n_triangles,), -1, dtype=torch.long)
        self.register_buffer('emitter_idx',emitter_idx)
        
        # emitter idx to triangle idx
        self.register_buffer('triangle_idx', torch.empty(0, 3))
        
        # randomly select a emitter
        self.register_buffer('emitter_pdf', torch.zeros(n_triangles, 3))
        self.register_buffer('emitter_cdf', torch.zeros(n_triangles, 3))

        # Cache for regularization
        self.caching = caching
        if self.caching:
            self.regularization_info = Batch(default=list)

        # Define the ativation
        self.activation = activation

    def load_state_dict(self, state_dict, *args, **kwargs):
        self.is_emitter = state_dict['is_emitter']
        self.emitter_vertices = state_dict['emitter_vertices']
        self.emitter_area = state_dict['emitter_area']
        self.emitter_normal = state_dict['emitter_normal']
        # self.radiance.data = torch.log(state_dict['radiance'].clamp(1e-12) * 100)
        self.radiance.data = state_dict['radiance']
        self.valid_emitter = state_dict['valid_emitter']

        self.emitter_idx = state_dict['emitter_idx']
        self.triangle_idx = state_dict['triangle_idx']

        self.emitter_pdf = state_dict['emitter_pdf']
        self.emitter_cdf = state_dict['emitter_cdf']

        # TODO: Remove this backward compatibility
        if 'original_is_emitter' in state_dict:
            self.original_is_emitter = state_dict['original_is_emitter']
        else:
            self.original_is_emitter = self.is_emitter.clone()

        self.slf.load_state_dict({k[4:]: v for k,v in state_dict.items() if k.startswith('slf.')}, *args, **kwargs)

        self.update_sampling()

    # def save(self, out_path):
    #     # Collate the pruned parameters
    #     emitter_keep_mask = self.is_emitter & self.original_is_emitter

    #     # Prune the parameters
    #     self.radiance = 

    def initialize(self, 
                   is_emitter,
                   emitter_vertices,
                   emitter_area,
                   emitter_normal,
                   emitter_radiance,
                   slf):
        self.is_emitter = is_emitter
        self.original_is_emitter = is_emitter.clone()
        self.emitter_vertices = emitter_vertices
        self.emitter_area = emitter_area
        self.emitter_normal = emitter_normal
        self.radiance.data = emitter_radiance
        self.valid_emitter = torch.ones_like(self.radiance.data[:, 0], dtype=torch.bool)
        self.slf = slf

        # emitter idx mapping, -1 indicates not an emitter
        emitter_idx = torch.full((len(is_emitter),),-1,device=is_emitter.device,dtype=torch.long)
        emitter_idx[is_emitter] = torch.arange(is_emitter.sum(),device=is_emitter.device)
        self.emitter_idx = emitter_idx
        
        # emitter idx to triangle idx
        triangle_idx = torch.arange(len(is_emitter))[is_emitter]
        self.triangle_idx = triangle_idx
        
        # Update Emitter Sampling
        self.update_sampling()

    @torch.no_grad()
    def update_sampling(self):
        # emitter_pdf = NF.normalize(self.emitter_area * self.radiance.sum(-1), dim=-1, p=1)
        emitter_pdf = NF.normalize(self.get_radiance().sum(-1), dim=-1, p=1)
        emitter_cdf = emitter_pdf.cumsum(-1).contiguous()
        self.emitter_pdf = emitter_pdf
        self.emitter_cdf = emitter_cdf
    
    def forward(self, position):
        """ surface light field from queried location """
        Le = self.slf(position)['rgb']
        return Le
    
    def eval_emitter(self, 
                     position, 
                     light_dir, 
                     triangle_idx,
                     roughness=None, 
                     trace_roughness=0.6):
        """ evaluate surface emission and pdf return radiance cache if diffuse
        Args:
            position: Bx3 intersection location
            light_dir: Bx3 emission direction
            triangle_idx: B intersected triangle id
            roughness: Bx1 surface roughness if not None
        Return:
            Le: Bx3 radiance
            emit_pdf: Bx1 emitter pdf
            valid_next: B valid surface
        """
        # whether valid intersection
        vis = triangle_idx != -1
        
        Le = torch.zeros(position.shape[0],3,device=position.device)
        emit_pdf = torch.zeros(position.shape[0],device=position.device)
        
        # get area light
        is_area = self.is_emitter[triangle_idx] & vis
        if is_area.any():
            # self.module_logger.debug(f"is_area {is_area}, triangle_idx {triangle_idx}, emitter_idx {self.emitter_idx}")
            # self.module_logger.debug(f"Evaluating emitters at {triangle_idx[is_area]}")
            e_idx = self.emitter_idx[triangle_idx[is_area]]
            # assert not (e_idx == -1).any(), f"Invalid emitter index found: e_idx {e_idx}, triangle_idx {triangle_idx[is_area]}, self.emitter_idx {self.emitter_idx}."
            emit_pdf[is_area] = self.emitter_pdf[e_idx]/self.emitter_area[e_idx].clamp_min(1e-12)

            # radiance = self.radiance[e_idx].clamp(0)
            radiance = self.get_radiance()[e_idx]
            Le[is_area] = radiance
            # Save the regularization info
            if self.caching and torch.is_grad_enabled():
                self.regularization_info["radiance"].append(radiance)
        
        # assume zero background lighting
        Le = Le*vis[...,None]
        valid_next = (~is_area)&vis

        # check diffuse radiance cache
        if isinstance(roughness, (int, float)) and roughness == -1:
            # for regularization purpose, always query the cache
            is_diffuse = (~is_area) & vis
            if is_diffuse.any():
                diffuse_slf = self.slf(position[is_diffuse])['rgb']
                L_diffuse = torch.zeros_like(Le)
                L_diffuse[is_diffuse] = diffuse_slf
                Le = Le + L_diffuse
                is_diffuse[is_diffuse.clone()] = diffuse_slf.sum(-1) > 0 # diffuse radiance need to > 0
                valid_next &= (~is_diffuse) # terminate path 
        elif roughness is not None:
            # query the radiance cache and terminate for diffuse and non emissive surface 
            is_diffuse = (~is_area) & vis & (roughness.squeeze(-1)>trace_roughness)
            if is_diffuse.any():
                diffuse_slf = self.slf(position[is_diffuse])['rgb']
                L_diffuse = torch.zeros_like(Le)
                L_diffuse[is_diffuse] = diffuse_slf
                Le = Le + L_diffuse
                is_diffuse[is_diffuse.clone()] = diffuse_slf.sum(-1) > 0 # diffuse radiance need to > 0
                valid_next &= (~is_diffuse) # terminate path 

        return Le, emit_pdf.unsqueeze(-1), valid_next, is_area
    

    def sample_emitter(self,sample1,sample2,position):
        """ importance sampling emitters
        Args:
            sample1: B uniform samples
            sample2: Bx2 uniform samples
            position: Bx3 surfae location
        Return:
            wi: Bx3 sampled direction
            pdf: Bx1 the sampling pdf (in area space)
            triangle_idx: B the sampled triangle id
        """
        # pick an emitter
        emitter_idx = torch.searchsorted(self.emitter_cdf,sample1)
        emitter_idx.clamp_(0, self.emitter_cdf.shape[0]-1)
        pdf0 = self.emitter_pdf[emitter_idx]

        # unifromly sample points on triangles
        xi1 = sample2[...,0].sqrt()
        u = (1-xi1).unsqueeze(-1)
        v = (xi1*sample2[...,1]).unsqueeze(-1)
        w = 1-u-v

        # emitter area
        A1 = self.emitter_area[emitter_idx]
        # sampled location on triangle
        p1 = self.emitter_vertices[emitter_idx]
        p1 = p1[:,0]*u + p1[:,1]*v + p1[:,2]*w
        wi = NF.normalize(p1-position,dim=-1)
        triangle_idx = self.triangle_idx[emitter_idx]
        
        # pdf in area space
        pdf = pdf0/A1.clamp_min(1e-12)
        return wi,pdf.unsqueeze(-1),triangle_idx

    def log_details(self, positions, directions, triangle_idx, b, h, w, spp, spp_batch, emission_gt=None):
        output = Batch()

        emission, _, _, emission_mask = batched_average(self.eval_emitter, 
                                                    Batch(position=einops.rearrange(positions, "(b spp) ... -> b spp ...", spp=spp), 
                                                            light_dir=einops.rearrange(directions, "(b spp) ... -> b spp ...", spp=spp), 
                                                            triangle_idx=einops.rearrange(triangle_idx, "(b spp) ... -> b spp ...", spp=spp)),
                                                    spp, spp_batch)
        emission_mask = emission_mask > 0.
        # emission, _, _, emission_mask = self.eval_emitter(positions, directions, triangle_idx)
        # emission = einops.rearrange(emission, "(b spp) c -> b spp c", spp=spp).mean(1)
        # emission_mask = einops.rearrange(emission_mask, "(b spp) -> b spp", spp=spp).any(dim=1)
        output['radiance'] = einops.rearrange(emission.clamp(0,1), '(b h w) c -> b c h w', b=b, h=h, w=w)
        output['mask'] = einops.rearrange(emission_mask.float(), '(b h w) -> b 1 h w', b=b, h=h, w=w)
        output['num_emitter_triangles'] = self.is_emitter.sum().float().unsqueeze(0)

        if emission_gt is not None:
            output["radiance_error"] = (einops.rearrange(emission - emission_gt, '(b h w) c -> b c h w', b=b, h=h, w=w) + 0.5).clamp(0,1)

        return output
    
    def get_regularization_loss(self):
        assert self.caching, "Regularization caching is disabled."

        # Retrieve from the cache
        if len(self.regularization_info["radiance"]) == 0:
            return Batch(monochrome=0., radiance=0.)

        emitter_radiance = torch.cat(self.regularization_info["radiance"], dim=0)

        # Clear the cache
        self.regularization_info = Batch(default=list)

        # Calculate regularization
        monochrome_regularization = emitter_radiance.std(dim=-1).mean()

        return Batch(
            monochrome=monochrome_regularization,
            radiance=emitter_radiance.mean()
        )

    def get_radiance(self):
        # return self.radiance.exp()
        if self.activation == "relu":
            return self.radiance.clamp(0.) * self.valid_emitter.unsqueeze(-1)
        elif self.activation == "exp":
            # Activation combining exp and relu to avoid vanishing gradient
            return (self.radiance.clamp(0.).exp() - 1.) * self.valid_emitter.unsqueeze(-1)
    
    @torch.no_grad()
    def prune_emitters(self, absolut_threshold=1., relative_threshold=0.1, percentile_threshold=0.1):
        """ Prune emitters based on their radiance values.
            Args:
                absolut_threshold: Absolute threshold for radiance values.
                relative_threshold: Relative threshold based on the mean radiance.
                percentile_threshold: Percentile threshold to prune the lowest emitters.
        """
        # Get current radiance values
        emitter_radiance = self.get_radiance().sum(dim=-1)

        # Determine thresholds
        thresholds = []
        if absolut_threshold is not None:
            thresholds.append(absolut_threshold)
        if relative_threshold is not None:
            thresholds.append(relative_threshold * emitter_radiance.max().item())
        if percentile_threshold is not None:
            thresholds.append(torch.quantile(emitter_radiance[emitter_radiance > 0], percentile_threshold).item())
        
        if len(thresholds) == 0:
            self.module_logger.info("No thresholds provided for pruning. Skipping pruning step.")
            return
        
        final_threshold = max(thresholds)
        # Identify emitters to keep
        emitter_keep_mask = (emitter_radiance > final_threshold)
        num_pruned = (~emitter_keep_mask).sum().item()
        remaining_emitters = emitter_keep_mask.sum().item()

        self.module_logger.info(f"Pruning {num_pruned} emitters from {emitter_radiance} (Threshold: {final_threshold:.4f})")
        self.module_logger.info(f"Remaining emitters: {remaining_emitters}")

        if remaining_emitters == 0:
            self.module_logger.warning("All emitters have been pruned. At least one emitter must remain.")
            return

        # Prune the parameters
        triangle_keep_mask = self.original_is_emitter.clone()
        triangle_keep_mask[self.original_is_emitter] = emitter_keep_mask
        self.is_emitter[~triangle_keep_mask] = False
        # self.emitter_vertices = self.emitter_vertices[emitter_keep_mask]
        # self.emitter_area = self.emitter_area[emitter_keep_mask]
        # self.emitter_normal = self.emitter_normal[emitter_keep_mask]

        # self.radiance.data[~emitter_keep_mask] = 0.  # Don't completely override here, because this is an optimizable parameter
        # self.radiance.data[~emitter_keep_mask] = -100.  # Don't completely override here, because this is an optimizable parameter
        self.valid_emitter = emitter_keep_mask

        # Update emitter_idx and triangle_idx
        # emitter_idx = torch.full((len(self.is_emitter),), -1, device=self.is_emitter.device, dtype=torch.long)
        # emitter_idx[self.is_emitter] = torch.arange(self.is_emitter.sum(), device=self.is_emitter.device)
        # self.emitter_idx = emitter_idx
        # triangle_idx = torch.arange(len(self.is_emitter), device=self.is_emitter.device)[self.is_emitter]
        # self.triangle_idx = triangle_idx

        self.emitter_idx[~triangle_keep_mask] = -1

        # # Update emitter_pdf and emitter_cdf
        # emitter_pdf = torch.ones_like(self.emitter_area)
        # self.emitter_pdf[~emitter_keep_mask] = 0.
        # emitter_pdf = NF.normalize(torch.ones_like(self.emitter_area), dim=-1, p=1)
        # emitter_cdf = emitter_pdf.cumsum(-1).contiguous()
        # self.emitter_pdf = emitter_pdf
        # self.emitter_cdf = emitter_cdf + (1 + 1e-12 - emitter_cdf[-1])  # Ensure the last value is slightly larger than 1.0

        self.update_sampling()
        # self.emitter_pdf[~emitter_keep_mask] = 0.
        # self.emitter_pdf = NF.normalize(self.emitter_pdf, dim=-1, p=1)
        # self.emitter_cdf = self.emitter_pdf.cumsum(-1).contiguous()
        


class SLFEnvmapImportanceEmitter(nn.Module):
    """ triangle emitters with diffuse radiance cache """
    def __init__(self, 
                 grid_size=1, 
                 n_triangles=0, 
                 n_emitters=0, 
                 envmap_resolution=(1024, 2048),
                 activation="relu", 
                 n_channels=3,
                 caching=True):
        """ 
        emitter_path: emitter parameter file
        slf_path: surface light field paramter file
        """
        super(SLFEnvmapImportanceEmitter,self).__init__()
        self.module_logger = init_logger()
        
        # load surface light field
        self.slf = VoxelSLF(grid_size=grid_size)
        
        # Define placeholder buffers
        self.register_buffer('is_emitter', torch.zeros(n_triangles, dtype=torch.bool))
        self.register_buffer('original_is_emitter', torch.zeros(n_triangles, dtype=torch.bool))
        self.register_buffer('emitter_vertices', torch.zeros(n_emitters, 3, 3))
        self.register_buffer('emitter_area', torch.zeros(n_emitters, 3))
        self.register_buffer('emitter_normal', torch.zeros(n_emitters, 3))
        self.register_buffer('n_channels', torch.tensor(n_channels, dtype=torch.long))

        # Define trainable parameters
        #  1. Mesh Emissions
        self.radiance = nn.Parameter(torch.zeros(n_emitters, n_channels))
        self.register_buffer('valid_emitter', torch.ones(n_emitters, dtype=torch.bool))
        self.register_buffer('num_emitters', torch.tensor(n_emitters, dtype=torch.long))

        #  2. Environment Map Emissions
        num_envmap_pixels = envmap_resolution[0] * envmap_resolution[1]
        self.envmap_radiance = nn.Parameter(torch.zeros(num_envmap_pixels, n_channels) + 1e-2)
        self.register_buffer('valid_envmap_pixel', torch.ones(num_envmap_pixels, dtype=torch.bool))
        self.register_buffer('num_envmap_pixels', torch.tensor(num_envmap_pixels, dtype=torch.long))
        self.register_buffer('envmap_resolution', torch.tensor(envmap_resolution, dtype=torch.long))
        
        # emitter idx mapping, -1 indicates not an emitter
        emitter_idx = torch.full((n_triangles,), -1, dtype=torch.long)
        self.register_buffer('emitter_idx',emitter_idx)
        
        # emitter idx to triangle idx
        self.register_buffer('triangle_idx', torch.empty(0, 3))
        
        # randomly select a emitter
        self.register_buffer('emitter_pdf', torch.zeros(n_triangles + num_envmap_pixels))
        self.register_buffer('emitter_cdf', torch.zeros(n_triangles + num_envmap_pixels))

        # Cache for regularization
        self.caching = caching
        if self.caching:
            self.regularization_info = Batch(default=list)

        # Define the ativation
        self.activation = activation

        self.update_sampling()

    def load_state_dict(self, state_dict, *args, **kwargs):
        self.is_emitter = state_dict['is_emitter']
        self.emitter_vertices = state_dict['emitter_vertices']
        self.emitter_area = state_dict['emitter_area']
        self.emitter_normal = state_dict['emitter_normal']
        
        self.radiance.data = state_dict['radiance']
        self.valid_emitter = state_dict['valid_emitter']
        self.num_emitters = state_dict['num_emitters']

        self.envmap_radiance.data = state_dict['envmap_radiance']
        self.valid_envmap_pixel = state_dict['valid_envmap_pixel']
        self.num_envmap_pixels = state_dict['num_envmap_pixels']
        self.envmap_resolution = state_dict['envmap_resolution']

        self.emitter_idx = state_dict['emitter_idx']
        self.triangle_idx = state_dict['triangle_idx']

        self.emitter_pdf = state_dict['emitter_pdf']
        self.emitter_cdf = state_dict['emitter_cdf']

        # TODO: Remove this backward compatibility
        if 'original_is_emitter' in state_dict:
            self.original_is_emitter = state_dict['original_is_emitter']
        else:
            self.original_is_emitter = self.is_emitter.clone()

        self.slf.load_state_dict({k[4:]: v for k,v in state_dict.items() if k.startswith('slf.')}, *args, **kwargs)

        self.update_sampling()

    # def save(self, out_path):
    #     # Collate the pruned parameters
    #     emitter_keep_mask = self.is_emitter & self.original_is_emitter

    #     # Prune the parameters
    #     self.radiance = 

    def initialize(self, 
                   is_emitter,
                   emitter_vertices,
                   emitter_area,
                   emitter_normal,
                   emitter_radiance,
                   slf,
                   envmap_radiance=None,
                   envmap_resolution=None,
                   valid_envmap_pixel=None):
        self.is_emitter = is_emitter
        self.original_is_emitter = is_emitter.clone()
        self.emitter_vertices = emitter_vertices
        self.emitter_area = emitter_area
        self.emitter_normal = emitter_normal

        self.radiance.data = emitter_radiance
        self.valid_emitter = torch.ones_like(self.radiance.data[:, 0], dtype=torch.bool)
        self.num_emitters = torch.tensor(len(emitter_radiance), dtype=torch.long)

        if envmap_radiance is not None:
            assert envmap_resolution is not None, "envmap_resolution must be provided when envmap_radiance is given."
            assert valid_envmap_pixel is not None, "valid_envmap_pixel must be provided when envmap_radiance is given."

            self.envmap_radiance.data = envmap_radiance
            self.valid_envmap_pixel = torch.tensor(valid_envmap_pixel, dtype=torch.bool)
            self.num_envmap_pixels = torch.tensor(len(envmap_radiance), dtype=torch.long)
            self.envmap_resolution = torch.tensor(envmap_resolution, dtype=torch.long)

        self.slf = slf

        # emitter idx mapping, -1 indicates not an emitter
        emitter_idx = torch.full((len(is_emitter),),-1,device=is_emitter.device,dtype=torch.long)
        emitter_idx[is_emitter] = torch.arange(is_emitter.sum(),device=is_emitter.device)
        self.emitter_idx = emitter_idx
        
        # emitter idx to triangle idx
        triangle_idx = torch.arange(len(is_emitter))[is_emitter]
        self.triangle_idx = triangle_idx
        
        # Update Emitter Sampling
        self.update_sampling()

    @torch.no_grad()
    def update_sampling(self):
        # Sampling strategy: 
        all_radiances = torch.cat([self.get_radiance().sum(-1), self.get_envmap_radiance().sum(-1)], dim=0)
        emitter_pdf = NF.normalize(all_radiances, dim=-1, p=1)
        emitter_cdf = emitter_pdf.cumsum(-1).contiguous()
        self.emitter_pdf = emitter_pdf
        self.emitter_cdf = emitter_cdf
    
    def forward(self, position):
        """ surface light field from queried location """
        Le = self.slf(position)['rgb']
        return Le
    
    def eval_emitter(self, 
                     position, 
                     light_dir, 
                     triangle_idx,
                     roughness=None, 
                     trace_roughness=0.6):
        """ evaluate surface emission and pdf return radiance cache if diffuse
        Args:
            position: Bx3 intersection location
            light_dir: Bx3 emission direction
            triangle_idx: B intersected triangle id
            roughness: Bx1 surface roughness if not None
        Return:
            Le: Bx3 radiance
            emit_pdf: Bx1 emitter pdf
            valid_next: B valid surface
        """
        # whether valid intersection
        vis = triangle_idx != -1
        
        Le = torch.zeros(position.shape[0],3,device=position.device)
        emit_pdf = torch.zeros(position.shape[0],device=position.device)
        
        # Mesh lighting
        is_area = self.is_emitter[triangle_idx] & vis
        if is_area.any():
            # self.module_logger.debug(f"is_area {is_area}, triangle_idx {triangle_idx}, emitter_idx {self.emitter_idx}")
            # self.module_logger.debug(f"Evaluating emitters at {triangle_idx[is_area]}")
            e_idx = self.emitter_idx[triangle_idx[is_area]]
            # assert not (e_idx == -1).any(), f"Invalid emitter index found: e_idx {e_idx}, triangle_idx {triangle_idx[is_area]}, self.emitter_idx {self.emitter_idx}."
            emit_pdf[is_area] = self.emitter_pdf[e_idx]/self.emitter_area[e_idx].clamp_min(1e-12)

            # radiance = self.radiance[e_idx].clamp(0)
            radiance = self.get_radiance()[e_idx]
            Le[is_area] = radiance
            # Save the regularization info
            if self.caching and torch.is_grad_enabled():
                self.regularization_info["radiance"].append(radiance)
        
        # Envmap lighting
        is_envmap = ~vis
        if is_envmap.any():
            envmap_dir = light_dir[is_envmap]
            envmap_pixel_idx = self.envmap_dir_to_idx(envmap_dir)

            emit_pdf[is_envmap] = self.emitter_pdf[self.num_emitters + envmap_pixel_idx]

            radiance = self.get_envmap_radiance()[envmap_pixel_idx]
            Le[is_envmap] = radiance

            # Save the regularization info
            if self.caching and torch.is_grad_enabled():
                if self.n_channels == 3:
                    self.regularization_info["radiance"].append(radiance)
                else:
                    self.regularization_info["radiance"].append(radiance.repeat(1, 3))

        # Next bounce goes only from non-emissive, visible surfaces
        valid_next = (~is_area) & vis

        # check diffuse radiance cache
        if isinstance(roughness, (int, float)) and roughness == -1:
            # for regularization purpose, always query the cache
            is_diffuse = (~is_area) & vis
            if is_diffuse.any():
                diffuse_slf = self.slf(position[is_diffuse])['rgb']
                L_diffuse = torch.zeros_like(Le)
                L_diffuse[is_diffuse] = diffuse_slf
                Le = Le + L_diffuse
                is_diffuse[is_diffuse.clone()] = diffuse_slf.sum(-1) > 0 # diffuse radiance need to > 0
                valid_next &= (~is_diffuse) # terminate path 
        elif roughness is not None:
            # query the radiance cache and terminate for diffuse and non emissive surface 
            is_diffuse = (~is_area) & vis & (roughness.squeeze(-1)>trace_roughness)
            if is_diffuse.any():
                diffuse_slf = self.slf(position[is_diffuse])['rgb']
                L_diffuse = torch.zeros_like(Le)
                L_diffuse[is_diffuse] = diffuse_slf
                Le = Le + L_diffuse
                is_diffuse[is_diffuse.clone()] = diffuse_slf.sum(-1) > 0 # diffuse radiance need to > 0
                valid_next &= (~is_diffuse) # terminate path 

        emission_mask = Le.sum(-1) > 0.

        return Le, emit_pdf.unsqueeze(-1), valid_next, emission_mask
    
    def envmap_dir_to_idx(self, directions):
        theta = torch.acos(directions[:,2].clamp(-1+1e-12, 1-1e-12))
        phi = torch.atan2(directions[:,1], directions[:,0])
        phi = phi % (2.0 * math.pi)

        theta_idx = (theta / math.pi * self.envmap_resolution[0]).long().clamp(0, self.envmap_resolution[0]-1)
        phi_idx = (phi / (2.0 * math.pi) * self.envmap_resolution[1]).long().clamp(0, self.envmap_resolution[1]-1)

        envmap_pixel_idx = theta_idx * self.envmap_resolution[1] + phi_idx
        return envmap_pixel_idx

    def sample_emitter(self,sample1,sample2,position):
        """ importance sampling emitters
        Args:
            sample1: B uniform samples
            sample2: Bx2 uniform samples
            position: Bx3 surfae location
        Return:
            wi: Bx3 sampled direction
            pdf: Bx1 the sampling pdf (in area space)
            triangle_idx: B the sampled triangle id
        """
        # pick an emitter
        emitter_idx = torch.searchsorted(self.emitter_cdf, sample1 * self.emitter_cdf[-1])
        emitter_idx.clamp_(0, self.emitter_cdf.shape[0]-1)
        pdf0 = self.emitter_pdf[emitter_idx]

        # Mask for envmap samples
        envmap_mask = emitter_idx >= self.num_emitters

        envmap_emitter_idx = emitter_idx[envmap_mask] - self.num_emitters
        mesh_emitter_idx = emitter_idx[~envmap_mask]

        mesh_sample2 = sample2[~envmap_mask]
        envmap_sample2 = sample2[envmap_mask]

        mesh_pdf0 = pdf0[~envmap_mask] 
        envmap_pdf0 = pdf0[envmap_mask]

        # Mesh samples
        # unifromly sample points on triangles
        xi1 = mesh_sample2[...,0].sqrt()
        u = (1-xi1).unsqueeze(-1)
        v = (xi1*mesh_sample2[...,1]).unsqueeze(-1)
        w = 1-u-v

        # emitter area
        A1 = self.emitter_area[mesh_emitter_idx]
        # sampled location on triangle
        p1 = self.emitter_vertices[mesh_emitter_idx]
        p1 = p1[:,0]*u + p1[:,1]*v + p1[:,2]*w
        mesh_position = position[~envmap_mask]
        wi_mesh = NF.normalize(p1-mesh_position,dim=-1)
        triangle_idx_mesh = self.triangle_idx[mesh_emitter_idx]
        
        # pdf in area space
        mesh_pdf0 = mesh_pdf0/A1.clamp_min(1e-12)


        # Envmap samples
        # Uniformly sample direction within the envmap pixel
        theta_center = envmap_emitter_idx // self.envmap_resolution[1]
        phi_center = envmap_emitter_idx % self.envmap_resolution[1]

        theta_idx = theta_center + envmap_sample2[...,0] - 0.5
        phi_idx = phi_center + envmap_sample2[...,1] - 0.5

        theta = (theta_idx / self.envmap_resolution[0]) * math.pi
        phi = (phi_idx / self.envmap_resolution[1]) * 2.0 * math.pi

        sin_theta = torch.sin(theta)
        x = sin_theta * torch.cos(phi)
        y = sin_theta * torch.sin(phi)
        z = torch.cos(theta)
        wi_envmap = torch.stack([x, y, z], dim=-1)
        triangle_idx_envmap = torch.zeros_like(envmap_emitter_idx) -1  # Indicate envmap samples with -1 triangle index

        # Pdf in world space
        envmap_pdf0 = torch.sin(theta_center) * ( (math.pi / self.envmap_resolution[0]) * (2.0 * math.pi / self.envmap_resolution[1]) )

        # Compose final outputs
        wi = torch.zeros_like(position)
        wi[~envmap_mask] = wi_mesh
        wi[envmap_mask] = wi_envmap

        pdf = torch.zeros_like(position[:,0])
        pdf[~envmap_mask] = mesh_pdf0
        pdf[envmap_mask] = envmap_pdf0

        triangle_idx = torch.zeros_like(position[:,0], dtype=torch.long)
        triangle_idx[~envmap_mask] = triangle_idx_mesh
        triangle_idx[envmap_mask] = triangle_idx_envmap

        return wi,pdf.unsqueeze(-1),triangle_idx

    def log_details(self, positions, directions, triangle_idx, b, h, w, spp, spp_batch, emission_gt=None):
        output = Batch()

        emission, _, _, emission_mask = batched_average(self.eval_emitter, 
                                                    Batch(position=einops.rearrange(positions, "(b spp) ... -> b spp ...", spp=spp), 
                                                            light_dir=einops.rearrange(directions, "(b spp) ... -> b spp ...", spp=spp), 
                                                            triangle_idx=einops.rearrange(triangle_idx, "(b spp) ... -> b spp ...", spp=spp)),
                                                    spp, spp_batch)
        emission_mask = emission_mask > 0.
        # emission, _, _, emission_mask = self.eval_emitter(positions, directions, triangle_idx)
        # emission = einops.rearrange(emission, "(b spp) c -> b spp c", spp=spp).mean(1)
        # emission_mask = einops.rearrange(emission_mask, "(b spp) -> b spp", spp=spp).any(dim=1)
        output['radiance'] = einops.rearrange(emission.clamp(0,1), '(b h w) c -> b c h w', b=b, h=h, w=w)
        output['mask'] = einops.rearrange(emission_mask.float(), '(b h w) -> b 1 h w', b=b, h=h, w=w)
        output['num_emitter_triangles'] = self.is_emitter.sum().float().unsqueeze(0)

        if emission_gt is not None:
            output["radiance_error"] = (einops.rearrange(emission - emission_gt, '(b h w) c -> b c h w', b=b, h=h, w=w) + 0.5).clamp(0,1)

        return output
    
    def get_regularization_loss(self):
        assert self.caching, "Regularization caching is disabled."

        # Retrieve from the cache
        if len(self.regularization_info["radiance"]) == 0:
            return Batch(monochrome=0., radiance=0.)

        emitter_radiance = torch.cat(self.regularization_info["radiance"], dim=0)

        # Clear the cache
        self.regularization_info = Batch(default=list)

        # Calculate regularization
        monochrome_regularization = emitter_radiance.std(dim=-1).mean()

        return Batch(
            monochrome=monochrome_regularization,
            radiance=emitter_radiance.mean()
        )

    def get_radiance(self):
        # return self.radiance.exp()
        if self.activation == "relu":
            return self.radiance.clamp(0.) * self.valid_emitter.unsqueeze(-1)
        elif self.activation == "exp":
            # Activation combining exp and relu to avoid vanishing gradient
            return (self.radiance.clamp(0.).exp() - 1.) * self.valid_emitter.unsqueeze(-1)

    def get_envmap_radiance(self):
        # return self.radiance.exp()
        if self.activation == "relu":
            return self.envmap_radiance.clamp(0.) * self.valid_envmap_pixel.unsqueeze(-1)
        elif self.activation == "exp":
            # Activation combining exp and relu to avoid vanishing gradient
            return (self.envmap_radiance.clamp(0.).exp() - 1.) * self.valid_envmap_pixel.unsqueeze(-1)
    
    @torch.no_grad()
    def prune_emitters(self, absolut_threshold=1., relative_threshold=0.1, percentile_threshold=0.1):
        """ Prune emitters based on their radiance values.
            Args:
                absolut_threshold: Absolute threshold for radiance values.
                relative_threshold: Relative threshold based on the mean radiance.
                percentile_threshold: Percentile threshold to prune the lowest emitters.
        """
        # Get current radiance values
        emitter_radiance = self.get_radiance().sum(dim=-1)
        envmap_radiance = self.get_envmap_radiance().sum(dim=-1)
        all_radiance = torch.cat([emitter_radiance, envmap_radiance], dim=0)

        # Determine thresholds
        thresholds = []
        if absolut_threshold is not None:
            thresholds.append(absolut_threshold)
        if relative_threshold is not None:
            thresholds.append(relative_threshold * all_radiance.max().item())
        if percentile_threshold is not None:
            thresholds.append(torch.quantile(all_radiance[all_radiance > 0], percentile_threshold).item())
        
        if len(thresholds) == 0:
            self.module_logger.info("No thresholds provided for pruning. Skipping pruning step.")
            return
        
        final_threshold = max(thresholds)

        # Identify emitters to keep
        emitter_keep_mask = (emitter_radiance > final_threshold)
        num_pruned = (~emitter_keep_mask).sum().item()
        remaining_emitters = emitter_keep_mask.sum().item()

        self.module_logger.info(f"Pruning {num_pruned} mesh emitters from {emitter_radiance} (Threshold: {final_threshold:.4f})")
        self.module_logger.info(f"Remaining mesh emitters: {remaining_emitters}")

        if remaining_emitters == 0:
            self.module_logger.warning("All emitters have been pruned. At least one emitter must remain.")
            return
        
        # Prune the parameters
        triangle_keep_mask = self.original_is_emitter.clone()
        triangle_keep_mask[self.original_is_emitter] = emitter_keep_mask
        self.is_emitter[~triangle_keep_mask] = False

        self.valid_emitter = emitter_keep_mask

        self.emitter_idx[~triangle_keep_mask] = -1

        # Identify envmap pixels to keep
        envmap_keep_mask = (envmap_radiance > final_threshold)
        num_pruned = (~envmap_keep_mask).sum().item()
        remaining_envmap_pixels = envmap_keep_mask.sum().item()

        self.module_logger.info(f"Pruning {num_pruned} envmap pixels from {envmap_radiance} (Threshold: {final_threshold:.4f})")
        self.module_logger.info(f"Remaining envmap pixels: {remaining_envmap_pixels}")

        if remaining_envmap_pixels == 0:
            self.module_logger.warning("All envmap pixels have been pruned. At least one pixel must remain.")
            return
        
        # Prune the parameters
        self.valid_envmap_pixel = envmap_keep_mask

        # Update emitter_pdf and emitter_cdf
        self.update_sampling()

        