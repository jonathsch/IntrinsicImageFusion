# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import einops
import torch
import torch.nn as nn
import torch.nn.functional as NF
import tinycudann as tcnn
import math

import sys

from iif.utils.batching import batched_average
from iif.utils.datastructure import Batch
from iif.utils.image_io import save_image
sys.path.append('..')

from iif.component.rendering.ops import *

EPS = 1e-20

def create_color_map(num_classes, seed=42):
    """Generate a color map for num_classes IDs."""
    g = torch.Generator().manual_seed(seed)  # deterministic colors
    colors = torch.randint(0, 256, (num_classes, 3), dtype=torch.uint8, generator=g) / 255
    return colors


def diffuse_sampler(sample2,normal):
    """ sampling diffuse lobe: wi ~ NoV/math.pi 
    Args:
        sample2: Bx2 uniform samples
        normal: Bx3 normal
    Return:
        wi: Bx3 sampled direction in world space
    """
    theta = torch.asin(sample2[...,0].sqrt())
    phi = math.pi*2*sample2[...,1]
    wi = angle2xyz(theta,phi)
    
    Nmat = get_normal_space(normal)
    wi = (wi[:,None]@Nmat.permute(0,2,1)).squeeze(1)    
    return wi

def specular_sampler(sample2,roughness,wo,normal):
    """ sampling ggx lobe: h ~ D/(VoH*4)*NoH
    Args:
        sample2: Bx3 uniform samples
        roughness: Bx1 roughness
        wo: Bx3 viewing direction
        normal: Bx3 normal
    Return:
        wi: Bx3 sampled direction in world space
    """
    alpha = (roughness*roughness).squeeze(-1).data
    
    # sample half vector
    theta = (1-sample2[...,0])/(sample2[...,0]*(alpha*alpha-1)+1)
    theta = torch.acos(theta.sqrt())
    phi = 2*math.pi*sample2[...,1]
    wh = angle2xyz(theta,phi)

    # half vector to wi
    Nmat = get_normal_space(normal)
    wh = (wh[:,None]@Nmat.permute(0,2,1)).squeeze(1)
    wi = 2*(wo*wh).sum(-1,keepdim=True)*wh-wo
    wi = NF.normalize(wi,dim=-1)
    return wi

class BaseBRDF(nn.Module):
    """ Base BRDF class """
    def __init__(self,):
        super(BaseBRDF,self).__init__()
        return
    
    def forward(self, position, **kwargs):
        # Return default material
        return {
            'albedo': torch.ones_like(position),
            'roughness': torch.ones_like(position[..., :1]),
            'metallic': torch.zeros_like(position[..., :1])
        }
    
    def eval_diffuse(self,wi,normal):
        """ evaluate diffuse shading 
            and pdf
        """
        pdf = (normal*wi).sum(-1,keepdim=True).relu()/math.pi
        brdf = pdf.expand(len(wi),3) 
        return brdf,pdf
    
    def sample_diffuse(self,sample2,normal):
        """ sample diffuse shading
            and get sampled weight
        """
        # get wi
        wi = diffuse_sampler(sample2,normal)
        
        # get brdf/pdf, pdf
        brdf_weight = torch.ones(normal.shape,device=normal.device)
        pdf = (normal*wi).sum(-1,keepdim=True).relu()/math.pi
        return wi,pdf,brdf_weight
    
    def eval_specular(self,wi,wo,normal,roughness):
        """" evaluate specular shadings
            and pdf
        """
        h = NF.normalize(wi+wo,dim=-1)
        NoL = (wi*normal).sum(-1,keepdim=True).relu()
        NoV = (wo*normal).sum(-1,keepdim=True).relu()
        VoH = (wo*h).sum(-1,keepdim=True).relu()
        NoH = (normal*h).sum(-1,keepdim=True).relu()

        D = D_GGX(NoH,roughness)
        pdf = D.data/(4*VoH.clamp_min(1e-4))*NoH

        G = G_Smith(NoV,NoL,roughness)
        F0,F1 = fresnelSchlick_sep(VoH)
        
        # two term corresponds to two fresnel components
        brdf_spec0 = D*G*F0/4.0*NoL
        brdf_spec1 = D*G*F1/4.0*NoL

        return brdf_spec0,brdf_spec1,pdf 

    def sample_specular(self,sample2,wo,normal,roughness):
        """ evaluate specular shadings
            and get sampled weight
        """
        # get wi
        wi = specular_sampler(sample2,roughness,wo,normal)
        
        # get brdf/pdf, pdf
        h = NF.normalize(wi+wo,dim=-1)
        NoL = (wi*normal).sum(-1,keepdim=True).relu()
        NoV = (wo*normal).sum(-1,keepdim=True).relu()
        VoH = (wo*h).sum(-1,keepdim=True).relu()
        NoH = (normal*h).sum(-1,keepdim=True).relu()
        
        D = D_GGX(NoH,roughness)
        pdf = D.data/(4*VoH.clamp_min(1e-4))*NoH

        G = G_Smith(NoV,NoL,roughness)
        F0,F1 = fresnelSchlick_sep(VoH)
        
        fac = G*VoH*NoL/NoH.clamp_min(1e-4)
        
        brdf_weight0 = F0*fac
        brdf_weight1 = F1*fac
        return wi,pdf,brdf_weight0,brdf_weight1
    
    def eval_brdf(self,wi,wo,normal,mat):
        """ evaluate BRDF and pdf
        Args:
            wi: Bx3 light direction
            wo: Bx3 viewing direction
            normal: Bx3 normal
            mat: surface BRDF dict
        Return:
            brdf: Bx3
            pdf: Bx1
        """
        albedo,roughness,metallic = mat['albedo'],mat['roughness'],mat['metallic']

        h = NF.normalize(wi+wo,dim=-1)
        NoL = (wi*normal).sum(-1,keepdim=True).relu()
        NoV = (wo*normal).sum(-1,keepdim=True).relu()
        VoH = (wo*h).sum(-1,keepdim=True).relu()
        NoH = (normal*h).sum(-1,keepdim=True).relu()

        # get pdf
        D = D_GGX(NoH,roughness)
        pdf_spec = D.data/(4*VoH.clamp_min(1e-4))*NoH
        pdf_diff = NoL/math.pi
        pdf = 0.5*pdf_spec + 0.5*pdf_diff

        # get brdf
        kd = albedo*(1-metallic)
        ks = 0.04*(1-metallic) + albedo*metallic

        G = G_Smith(NoV,NoL,roughness)
        F = fresnelSchlick(VoH,ks)
        brdf_diff = kd/math.pi*NoL
        brdf_spec = D*G*F/4.0*NoL

        brdf = brdf_diff + brdf_spec

        return brdf,pdf 
    
    def sample_brdf(self,sample1,sample2,wo,normal,mat):
        """ importance sampling brdf and get brdf/pdf
        Args:
            sample1: B unifrom samples
            sample2: Bx2 uniform samples
            wo: Bx3 viewing direction
            normal: Bx3 normal
            mat: material dict
        Return:
            wi: Bx3 sampled direction
            pdf: Bx1
            brdf_weight: Bx3 brdf/pdf
        """
        B = sample1.shape[0]
        device = sample1.device

        pdf = torch.zeros(B,device=device)
        brdf = torch.zeros(B,3,device=device)
        wi = torch.zeros(B,3,device=device)

        mask = (sample1 > 0.5)
        # sample diffuse
        wi[mask] = diffuse_sampler(sample2[mask],normal[mask])
        mask = ~mask
        # sample specular
        wi[mask] = specular_sampler(sample2[mask],mat['roughness'][mask],wo[mask],normal[mask])

        # get brdf,pdf
        brdf,pdf = self.eval_brdf(wi,wo,normal,mat)

        brdf_weight = torch.where(pdf>0,brdf/pdf.clamp(EPS),0)
        brdf_weight[brdf_weight.isnan()] = 0
        return wi,pdf,brdf_weight
    
class NGPSegmentation(BaseBRDF):
    """ Hash Grid based brdf paramterization """
    def __init__(self,
                 voxel_min=0,
                 voxel_max=1,
                 num_classes=100):
        """ 
        voxel_min,voxel_max: scene bounding box
        """
        super(NGPSegmentation,self).__init__()
        
        self.register_buffer('voxel_min', torch.tensor(voxel_min))
        self.register_buffer('voxel_max', torch.tensor(voxel_max))
        self.register_buffer('num_classes', torch.tensor(num_classes))

        hash_encoding={
                "otype": "HashGrid",
                # "n_levels": 16,
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
            n_output_dims=self.num_classes, 
            encoding_config=hash_encoding, 
            network_config=hash_network)

        # self.encoding = tcnn.Encoding(
        #     n_input_dims=3, 
        #     encoding_config=hash_encoding
        # )
        # # network = tcnn.Network(
        # #     n_input_dims=encoding.n_output_dims, 
        # #     n_output_dims=self.num_classes, 
        # #     network_config=hash_network
        # # )
        # # self.mlp = torch.nn.Sequential(encoding, network)
        # self.network = nn.Linear(self.encoding.n_output_dims, self.num_classes)
        # # self.mlp = torch.nn.Sequential(encoding, nn.Linear(encoding.n_output_dims, self.num_classes))

        # # Init with small values
        # with torch.no_grad():
        #     for param in encoding.parameters():
        #         param *= 0.
        #         param += 1e-2
        
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
        
        # features = self.encoding(position*2-1).float()
        # segmentation = self.network(features)
        segmentation = self.mlp(position*2-1)
        return segmentation.float()
    
    
class NGPBRDF(BaseBRDF):
    """ Hash Grid based brdf paramterization """
    def __init__(self,
                 voxel_min=0,
                 voxel_max=1):
        """ 
        voxel_min,voxel_max: scene bounding box
        """
        super(NGPBRDF,self).__init__()

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
            n_output_dims=5, 
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
        
        mat = self.mlp(position*2-1).sigmoid()
        return {
            'albedo': mat[...,:3].float(),
            'roughness': mat[...,3:4].float()*0.98+0.02, # avoid nan
            'metallic': mat[...,4:5].float()
        }

    def log_details(self, positions, b, h, w, spp, spp_batch, mask=None, **kwargs):
        # Collect the predictions to visualize
        mat = batched_average(self, einops.rearrange(positions, "(b spp) ... -> b spp ...", spp=spp), spp, spp_batch)
        # mat = self(positions)
        output = Batch(
            albedo=mat["albedo"],
            roughness=mat["roughness"],
            metallic=mat["metallic"]
        )
        # output = output.map(lambda x: einops.rearrange(x, "(b spp) c -> b spp c", spp=spp).mean(1))

        # Apply masking
        if mask is not None:
            output = output * mask.unsqueeze(-1)

        # Reshape the predictions
        output = output.map(lambda x: einops.rearrange(x, '(b h w) c -> b c h w', b=b, h=h, w=w))

        return output
    
    
class NGPAssignment(BaseBRDF):
    """ Hash Grid based brdf paramterization """
    def __init__(self,
                 voxel_min=0,
                 voxel_max=1):
        """ 
        voxel_min,voxel_max: scene bounding box
        """
        super(NGPAssignment,self).__init__()

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
        
        assignment = self.mlp(position*2-1)
        return assignment.float()
    
class ProbabilisticNGPBRDF(BaseBRDF):
    """ Hash Grid based brdf paramterization """
    def __init__(self,
                 voxel_min=0,
                 voxel_max=1):
        """ 
        voxel_min,voxel_max: scene bounding box
        """
        super(ProbabilisticNGPBRDF,self).__init__()

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
            n_output_dims=8, 
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
        
        mat = self.mlp(position*2-1).sigmoid()
        # return {
        #     'albedo': mat[...,:3].float(),
        #     'albedo_std': mat[...,3:6].float(),
        #     'roughness': 0 * mat[...,6:7].float() + 1, # avoid nan
        #     'metallic': 0 * mat[...,7:8].float() + 0
        # }
        return {
            'albedo': mat[...,:3].float(),
            'albedo_std': mat[...,3:6].float(),
            'roughness': mat[...,6:7].float()*0.98+0.02, # avoid nan
            'metallic': mat[...,7:8].float()
        }
    
    def log_details(self, mat, b, h, w, **kwargs):
        # Collect the predictions to visualize
        output = Batch(
            albedo=mat["albedo"],
            roughness=mat["roughness"],
            metallic=mat["metallic"]
        )
        # output = output.map(lambda x: einops.rearrange(x, "(b spp) c -> b spp c", spp=spp).mean(1))

        # Reshape the predictions
        output = output.map(lambda x: einops.rearrange(x, '(b h w) c -> b c h w', b=b, h=h, w=w))

        return output

class TextureBRDF(BaseBRDF):
    """ Textured mesh based brdf parameterization (unsued) """
    def __init__(self,res):
        super(TextureBRDF,self).__init__()
        self.res = res
        self.register_parameter('textures',nn.Parameter(torch.randn(self.res,self.res,3+2)*1e-2))
        
    def forward(self,uv):
        uv = (uv*(self.res-1)).clamp(0,self.res-1)
        uv0 = uv.floor().long()
        uv1 = uv.ceil().long()
        uv_ = uv-uv0
        
        u0,v0 = uv0.T
        u1,v1 = uv1.T
 
        
        t00,t01 = self.textures[v0,u0].sigmoid(),self.textures[v0,u1].sigmoid()
        t10,t11 = self.textures[v1,u0].sigmoid(),self.textures[v1,u1].sigmoid()
        
        u_,v_ = uv_.T
        u_,v_ = u_.unsqueeze(-1),v_.unsqueeze(-1)
        
        t = t00*(1-u_)*(1-v_) + t01*u_*(1-v_)\
          + t10*(1-u_)*v_ + t11*u_*v_
        
        return {
            'albedo': t[...,:3],
            'roughness': t[...,3:4]*0.98+0.02, # avoid nan
            'metallic': t[...,4:5]
        }  
    

class ObjTransformedBRDF(BaseBRDF):
    """ Hash Grid based brdf paramterization """
    def __init__(self,
                 brdf_net=None,
                 semantic_net=None):
        """ 
        voxel_min,voxel_max: scene bounding box
        """
        super(ObjTransformedBRDF,self).__init__()
        self.brdf_net = brdf_net
        self.semantic_net = semantic_net
        self.albedo_transform = torch.nn.Parameter(torch.tensor([[1., 0., 0., 0.],
                                                                [0., 1., 0., 0.],
                                                                [0., 0., 1., 0.]])[None, ...].repeat(self.semantic_net.num_classes, 1, 1))
        
        self.seen_transforms = []
        
    def forward(self, position, triangle_idx, **kwargs):
        """ query brdf parameters at given location
        Args:
            position: Bx3 queried location
        Return:
            Bx3 base color
            Bx1 roughness in [0.02,1]
            Bx1 metallic
        """
        with torch.no_grad():
            # Eval material
            mat = self.brdf_net(position)

            # Eval semantics
            segmentation = self.semantic_net(position=position, triangle_idx=triangle_idx)
            if segmentation.ndim == 2:
                segmentation = segmentation.argmax(-1)
            segmentation = segmentation.long()

        # Apply transform
        albedo = mat['albedo']
        albedo_transform = self.albedo_transform[segmentation]
        albedo = einops.einsum(torch.cat([albedo, torch.ones_like(albedo[..., :1])], dim=-1), albedo_transform, "B D, B C D -> B C")
        mat['albedo'] = albedo

        # Save seen transforms
        self.seen_transforms.append(albedo_transform)

        return mat
    
    def get_seen_transforms(self):
        seen_transforms = torch.cat(self.seen_transforms, dim=0)
        self.seen_transforms = []
        return seen_transforms
    









# ========================== Ours ==========================
class ProbabilisticBRDF(BaseBRDF):
    """ Hash Grid based brdf paramterization """
    def __init__(self,
                 voxel_min=0,
                 voxel_max=1):
        """ 
        voxel_min,voxel_max: scene bounding box
        """
        super(ProbabilisticBRDF,self).__init__()

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
            n_output_dims=10, 
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
        # Map to [0,1]
        position = (position-self.voxel_min)/(self.voxel_max-self.voxel_min)
        
        mat = self.mlp(position*2-1).sigmoid()

        return {
            'albedo': mat[...,:3].float(),
            'albedo_std': mat[...,3:6].float(),

            'roughness': mat[...,6:7].float()*0.98+0.02, # avoid nan
            'roughness_std': mat[...,7:8].float(),

            'metallic': mat[...,8:9].float(),
            'metallic_std': mat[...,9:10].float()
        }
    
    def log_details(self, mat, b, h, w, mask=None):
        # Collect the predictions to visualize
        output = Batch(
            albedo=mat["albedo"],
            albedo_std=mat["albedo_std"],

            roughness=mat["roughness"],
            roughness_std=mat["roughness_std"],

            metallic=mat["metallic"],
            metallic_std=mat["metallic_std"],
        )
        # output = output.map(lambda x: einops.rearrange(x, "(b spp) c -> b spp c", spp=spp).mean(1))

        # Apply masking
        if mask is not None:
            output = output * mask.unsqueeze(-1)

        # Reshape the predictions
        output = output.map(lambda x: einops.rearrange(x, '(b h w) c -> b c h w', b=b, h=h, w=w))

        return output
    

class TransformedBRDF(BaseBRDF):
    def __init__(self,
                 brdf_base,
                 brdf_transform):
        super(TransformedBRDF,self).__init__()
        self.brdf_base = brdf_base
        self.brdf_transform = brdf_transform

        self.reg_cache = Batch(default=list)

    def forward(self, position, triangle_idx, caching=True):
        # Eval material
        mat = self.brdf_base(position)

        # Apply transform
        mat = self.brdf_transform(position=position, triangle_idx=triangle_idx, mat=mat, caching=caching)

        # Save to cache for regularization
        if caching and torch.is_grad_enabled():
            self.reg_cache["mat"].append(mat)

        return mat
    
    def log_details(self, position, triangle_idx, spp, spp_batch, *args, **kwargs):
        def get_predictions(position, triangle_idx):
            base = self.brdf_base(position)
            transformed = self.brdf_transform(position=position, triangle_idx=triangle_idx, mat=base, caching=False)
            return base, transformed

        # Eval material
        base, transformed = batched_average(get_predictions, Batch(position=einops.rearrange(position, "(b spp) ... -> b spp ...", spp=spp),
                                                                   triangle_idx=einops.rearrange(triangle_idx, "(b spp) ... -> b spp ...", spp=spp)), 
                                                                   spp, spp_batch)
        base_logs = self.brdf_base.log_details(base, *args, **kwargs)

        # Apply transform
        # transformed = self.brdf_transform(position, base)
        transformation_logs = self.brdf_transform.log_details(*args, position=position, triangle_idx=triangle_idx, base=base, spp=spp, spp_batch=spp_batch, **kwargs)
        transformed_logs = self.brdf_base.log_details(transformed, *args, **kwargs)

        return Batch(
            base=base_logs,
            transformed=transformed_logs,
            transformation=transformation_logs
        )
    
    def get_regularization_loss(self):
        regularizations = {
            "transform": self.brdf_transform.get_regularization_loss()
        }

        # Calculate diffuse regularization
        if len(self.reg_cache["mat"]) > 0:
            mat = self.reg_cache["mat"]
            roughness = torch.cat([m['roughness'] for m in mat], dim=0)
            metallic = torch.cat([m['metallic'] for m in mat], dim=0)

            # regularizations["roughness"] = ((roughness-1)**2).mean()
            # regularizations["metallic"] = (metallic**2).mean()
            regularizations["roughness"] = (roughness-1).abs().mean()
            regularizations["metallic"] = metallic.abs().mean()

            # Clear the cache
            self.reg_cache = Batch(default=list)

        return regularizations
    

class LinearObjAlbedoTransform(nn.Module):
    def __init__(self, 
                 semantic_net):
        super(LinearObjAlbedoTransform,self).__init__()
        self.semantic_net = semantic_net
        self.albedo_transform = torch.nn.Parameter(torch.tensor([[1., 0., 0., 0.],
                                                                [0., 1., 0., 0.],
                                                                [0., 0., 1., 0.]])[None, ...].repeat(self.semantic_net.num_classes, 1, 1))
        

        self.regularization_info = Batch(default=list)
        self.register_buffer('color_map', create_color_map(self.semantic_net.num_classes))

    def forward(self, position, triangle_idx, mat):
        # Eval semantics
        segmentation = self.semantic_net(position=position, triangle_idx=triangle_idx)
        if segmentation.ndim == 2:
            segmentation = segmentation.argmax(-1)
        segmentation = segmentation.long()

        # Apply transform
        albedo = mat['albedo']
        albedo_transform = self.albedo_transform[segmentation]
        albedo = einops.einsum(torch.cat([albedo, torch.ones_like(albedo[..., :1])], dim=-1), albedo_transform, "B D, B C D -> B C")
        mat['albedo'] = albedo

        # Save the regularization info
        self.regularization_info["albedo_transform"].append(albedo_transform)

        return mat
    
    def log_details(self, position, triangle_idx, base, b, h, w, spp, spp_batch, **kwargs):
        # Eval semantics
        segmentation = batched_average(self.semantic_net, Batch(position=einops.rearrange(position, "(b spp) ... -> b spp ...", spp=spp),
                                                                triangle_idx=einops.rearrange(triangle_idx, "(b spp) ... -> b spp ...", spp=spp)), 
                                                                spp, spp_batch)
        if segmentation.ndim == 2:
            segmentation = segmentation.argmax(-1)
        segmentation = segmentation.unsqueeze(-1).long()

        # Create a color-coded segmentation map
        # segmentation = self.semantic_net(position).argmax(-1).unsqueeze(-1).long()
        # segmentation = einops.rearrange(segmentation, "(b spp) c -> b spp c", spp=spp).mean(1)
        hard_assignments = NF.one_hot(segmentation, num_classes=self.semantic_net.num_classes).float()
        segmentation = (self.color_map[None,None,...] * hard_assignments[...,None]).sum(dim=-2).squeeze(1)

        return Batch(
            segmentation=einops.rearrange(segmentation, '(b h w) c -> b c h w', b=b, h=h, w=w),
        )
    
    def get_regularization_loss(self):
        albedo_transforms = torch.cat(self.regularization_info["albedo_transform"], dim=0)

        self.regularization_info = Batch(default=list)

        A, b = albedo_transforms[..., :3], albedo_transforms[..., 3] 
        Sigma = torch.eye(3, device=A.device)[None, ...]
        I = torch.eye(3, device=A.device)[None, ...]
        loss_albedo_transform = (((A - I) ** 2).sum(dim=(-2, -1)) + (b ** 2).sum(dim=-1) + ((A @ Sigma @ A.transpose(-1, -2) - Sigma) ** 2).sum(dim=(-2, -1))).mean()

        return loss_albedo_transform



class LinearObjBRDFTransform(nn.Module):
    def __init__(self, 
                 semantic_net,
                 caching=True):
        super(LinearObjBRDFTransform,self).__init__()
        self.semantic_net = semantic_net
        self.albedo_transform = torch.nn.Parameter(torch.tensor([[1., 0., 0., 0.],
                                                                [0., 1., 0., 0.],
                                                                [0., 0., 1., 0.]])[None, ...].repeat(self.semantic_net.num_classes, 1, 1))
        self.roughness_transform = torch.nn.Parameter(torch.tensor([[1., 0.]])[None, ...].repeat(self.semantic_net.num_classes, 1, 1))
        self.metallic_transform = torch.nn.Parameter(torch.tensor([[1., 0.]])[None, ...].repeat(self.semantic_net.num_classes, 1, 1))        

        self.regularization_info = Batch(default=list)
        self.register_buffer('color_map', create_color_map(self.semantic_net.num_classes))

    def forward(self, position, triangle_idx, mat, caching=True):
        # Eval semantics
        segmentation = self.semantic_net(position=position, triangle_idx=triangle_idx)
        if segmentation.ndim == 2:
            segmentation = segmentation.argmax(-1)
        segmentation = segmentation.long()

        # Apply transform
        albedo = mat['albedo']
        albedo_transform = self.albedo_transform[segmentation]
        albedo = einops.einsum(torch.cat([albedo, torch.ones_like(albedo[..., :1])], dim=-1), albedo_transform, "B D, B C D -> B C")
        mat['albedo'] = albedo.clamp(0,1)

        roughness = mat['roughness']
        roughness_transform = self.roughness_transform[segmentation]
        roughness = einops.einsum(torch.cat([roughness, torch.ones_like(roughness)], dim=-1), roughness_transform, "B D, B C D -> B C")
        mat['roughness'] = roughness.clamp(0.02,1)

        metallic = mat['metallic']
        metallic_transform = self.metallic_transform[segmentation]  
        metallic = einops.einsum(torch.cat([metallic, torch.ones_like(metallic)], dim=-1), metallic_transform, "B D, B C D -> B C")
        mat['metallic'] = metallic.clamp(0,1)

        # Save the regularization info
        if caching and torch.is_grad_enabled():
            self.regularization_info["albedo_transform"].append(albedo_transform)
            self.regularization_info["roughness_transform"].append(roughness_transform)
            self.regularization_info["metallic_transform"].append(metallic_transform)

        return mat
    
    def log_details(self, position, triangle_idx, base, b, h, w, spp, spp_batch, **kwargs):
        # Eval semantics
        segmentation = batched_average(self.semantic_net, Batch(position=einops.rearrange(position, "(b spp) ... -> b spp ...", spp=spp),
                                                                triangle_idx=einops.rearrange(triangle_idx, "(b spp) ... -> b spp ...", spp=spp)), 
                                                                spp, spp_batch)
        if segmentation.ndim == 2:
            segmentation = segmentation.argmax(-1)
        segmentation = segmentation.unsqueeze(-1).long()

        # Create a color-coded segmentation map
        # segmentation = self.semantic_net(position).argmax(-1).unsqueeze(-1).long()
        # segmentation = einops.rearrange(segmentation, "(b spp) c -> b spp c", spp=spp).float().mean(1).long()
        hard_assignments = NF.one_hot(segmentation, num_classes=self.semantic_net.num_classes).float()
        segmentation = (self.color_map[None,None,...] * hard_assignments[...,None]).sum(dim=-2).squeeze(1)

        return Batch(
            segmentation=einops.rearrange(segmentation, '(b h w) c -> b c h w', b=b, h=h, w=w),
        )
    
    def get_regularization_loss(self):
        # Albedo
        albedo_transforms = torch.cat(self.regularization_info["albedo_transform"], dim=0)
        A, b = albedo_transforms[..., :3], albedo_transforms[..., 3] 
        Sigma = torch.eye(3, device=A.device)[None, ...]
        I = torch.eye(3, device=A.device)[None, ...]
        loss_albedo_transform = (((A - I) ** 2).sum(dim=(-2, -1)) + (b ** 2).sum(dim=-1) + ((A @ Sigma @ A.transpose(-1, -2) - Sigma) ** 2).sum(dim=(-2, -1))).mean()

        # Roughness
        roughness_transforms = torch.cat(self.regularization_info["roughness_transform"], dim=0)
        loss_roughness_transform = (((roughness_transforms[..., 0] - 1) ** 2) + (roughness_transforms[..., 1] ** 2)).mean()

        # Metallic
        metallic_transforms = torch.cat(self.regularization_info["metallic_transform"], dim=0)
        loss_metallic_transform = (((metallic_transforms[..., 0] - 1) ** 2) + (metallic_transforms[..., 1] ** 2)).mean()

        # Clear the cache
        self.regularization_info = Batch(default=list)

        return {
            "albedo": loss_albedo_transform,
            "roughness": loss_roughness_transform,
            "metallic": loss_metallic_transform
        }

class LinearObjImageBRDFTransform(nn.Module):
    def __init__(self, 
                 num_segments, 
                 num_predictions_per_image):
        super(LinearObjImageBRDFTransform,self).__init__()
        self.num_segments = num_segments
        self.num_predictions_per_image = num_predictions_per_image
        self.albedo_transform = torch.nn.Parameter(torch.tensor([[1., 0., 0., 0.],
                                                                [0., 1., 0., 0.],
                                                                [0., 0., 1., 0.]])[None, None].repeat(self.num_segments, self.num_predictions_per_image, 1, 1))
        self.roughness_transform = torch.nn.Parameter(torch.tensor([[1., 0.]])[None, None].repeat(self.num_segments, self.num_predictions_per_image, 1, 1))
        self.metallic_transform = torch.nn.Parameter(torch.tensor([[1., 0.]])[None, None].repeat(self.num_segments, self.num_predictions_per_image, 1, 1))

        self.regularization_info = Batch(default=list)

    def forward(self, segmentation, materials):
        transformed_materials = Batch()

        assert segmentation.min() >= 0 and segmentation.max() < self.num_segments, f"Segmentation indices out of range: {segmentation}"

        # Apply transform
        albedo = materials['albedo']
        albedo_transform = self.albedo_transform[segmentation]
        albedo = einops.einsum(torch.cat([albedo, torch.ones_like(albedo[..., :1])], dim=-1), albedo_transform, "B P D, B P C D -> B P C")
        transformed_materials['albedo'] = albedo.clamp(0, 1)

        roughness = materials['roughness']
        roughness_transform = self.roughness_transform[segmentation]
        roughness = einops.einsum(torch.cat([roughness, torch.ones_like(roughness)], dim=-1), roughness_transform, "B P D, B P C D -> B P C")
        transformed_materials['roughness'] = roughness.clamp(0.02, 1)

        metallic = materials['metallic']
        metallic_transform = self.metallic_transform[segmentation]  
        metallic = einops.einsum(torch.cat([metallic, torch.ones_like(metallic)], dim=-1), metallic_transform, "B P D, B P C D -> B P C")
        transformed_materials['metallic'] = metallic.clamp(0, 1)

        # Save the regularization info
        self.regularization_info["albedo_transform"].append(albedo_transform)
        self.regularization_info["roughness_transform"].append(roughness_transform)
        self.regularization_info["metallic_transform"].append(metallic_transform)

        return transformed_materials
    
    def log_details(self, position, base, b, h, w):
        raise NotImplementedError()
    
    def get_regularization_loss(self):
        # Albedo
        albedo_transforms = torch.cat(self.regularization_info["albedo_transform"], dim=0)
        A, b = albedo_transforms[..., :3], albedo_transforms[..., 3] 
        Sigma = torch.eye(3, device=A.device)[None, ...]
        I = torch.eye(3, device=A.device)[None, ...]
        loss_albedo_transform = (((A - I) ** 2).sum(dim=(-2, -1)) + (b ** 2).sum(dim=-1) + ((A @ Sigma @ A.transpose(-1, -2) - Sigma) ** 2).sum(dim=(-2, -1))).mean()

        # Roughness
        roughness_transforms = torch.cat(self.regularization_info["roughness_transform"], dim=0)
        loss_roughness_transform = (((roughness_transforms[..., 0] - 1) ** 2) + (roughness_transforms[..., 1] ** 2)).mean()

        # Metallic
        metallic_transforms = torch.cat(self.regularization_info["metallic_transform"], dim=0)
        loss_metallic_transform = (((metallic_transforms[..., 0] - 1) ** 2) + (metallic_transforms[..., 1] ** 2)).mean()

        # Clear the cache
        self.regularization_info = Batch(default=list)

        return {
            "albedo": loss_albedo_transform,
            "roughness": loss_roughness_transform,
            "metallic": loss_metallic_transform
        }

