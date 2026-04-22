# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as NF

""" surface light field model """
        

class VoxelSLF(nn.Module):
    """ voxel grid based surface light field """
    def __init__(self, grid_size=1):
        """
        mask: NxNxN voxel occupancy mask
        voxel_min,voxel_max: voxel bounding box
        """
        super(VoxelSLF,self).__init__()

        self.grid_size = grid_size

        # Define the voxel grid
        self.register_buffer("mask", torch.zeros(grid_size, grid_size, grid_size))
        self.register_buffer("voxel_min", torch.tensor(0.))
        self.register_buffer("voxel_max", torch.tensor(0.))

        # Define the sparse voxel grid, storing radiance values
        self.register_buffer('inds', -torch.ones(grid_size, grid_size, grid_size, dtype=torch.long))
        self.register_buffer('radiance', torch.tensor(0.))
        self.register_buffer('count', torch.tensor(0)) # number of entries, used for mean pooling

    def load_state_dict(self, state_dict, *args, **kwargs):
        # Load the sparse voxel grid manually
        self.radiance = state_dict['radiance']
        self.count = state_dict['count']
        
        super().load_state_dict(state_dict)

    def initialize(self, mask, voxel_min, voxel_max):
        self.grid_size = mask.shape[0]
        self.mask = mask
        self.voxel_min = torch.tensor(voxel_min)
        self.voxel_max = torch.tensor(voxel_max)

        kk,jj,ii = torch.where(mask)
        self.inds = -torch.ones(self.grid_size, self.grid_size, self.grid_size, dtype=torch.long)
        self.inds[kk,jj,ii] = torch.arange(len(ii))
        self.radiance = torch.zeros(len(ii),3)*1e-1
        self.count = torch.zeros(len(ii),dtype=torch.long)
        return self

    def spatial_idx(self,x):
        """ get voxel entry index for input location
        Args:
            x: Bx3 3D position
        Return:
            B indices
        """
        # map to voxel grid coordinates
        x_ = (x-self.voxel_min)/(self.voxel_max-self.voxel_min)
        x_ = (x_*self.grid_size).long().clamp(0,self.grid_size-1)

        # find entry indices
        assert torch.all(x_<self.grid_size) and torch.all(x_<self.grid_size), f"Spatial index out of bounds: {x_}"
        idx = self.inds[x_[...,2],x_[...,1],x_[...,0]]
        return idx
        
    def scatter_add(self,x,radiance):
        """ scatter add radiance to voxel grid
        """
        idx = self.spatial_idx(x)
        self.radiance.scatter_add_(0,idx[...,None].expand_as(radiance),radiance)
        self.count.scatter_add_(0,idx,torch.ones_like(idx))

    def compute(self):
        """ average pooling the radiance """
        self.radiance = self.radiance / self.count[...,None].float().clamp_min(1)
        self.count = torch.ones_like(self.count)
        return self
    
    def forward(self,x):
        """ query surface light field """
        idx = self.spatial_idx(x)
        radiance = self.radiance[idx]
        radiance[idx==-1] = 0 # if hit empty space, return zero radiance
        return {
            'rgb': radiance
        }

        
class VoxelSLF_old(nn.Module):
    """ voxel grid based surface light field """
    def __init__(self, grid_size):
        """
        mask: NxNxN voxel occupancy mask
        voxel_min,voxel_max: voxel bounding box
        """
        super(VoxelSLF,self).__init__()

        mask = mask
        voxel_min = torch.tensor(voxel_min)
        voxel_max = torch.tensor(voxel_max)

        # find coordinates for occupied voxels
        kk,jj,ii = torch.where(mask)
        inds = -torch.ones(grid_size, grid_size, grid_size, dtype=torch.long)
        inds[kk,jj,ii] = torch.arange(len(ii))

        radiance = torch.zeros(len(ii),3)*1e-1
        count = torch.zeros(len(ii),dtype=torch.long)

        self.register_buffer("H", H)
        self.register_buffer("mask", mask)
        self.register_buffer("voxel_min", voxel_min)
        self.register_buffer("voxel_max", voxel_max)
        self.register_buffer('inds', inds)
        self.register_buffer('radiance', radiance)
        self.register_buffer('count', count) # number of entries, used for mean pooling

    def spatial_idx(self,x):
        """ get voxel entry index for input location
        Args:
            x: Bx3 3D position
        Return:
            B indices
        """
        # map to voxel grid coordinates
        x_ = (x-self.voxel_min)/(self.voxel_max-self.voxel_min)
        x_ = (x_*self.H).long().clamp(0,self.H-1)

        # find entry indices
        idx = self.inds[x_[...,2],x_[...,1],x_[...,0]]
        return idx
        
    def scatter_add(self,x,radiance):
        """ scatter add radiance to voxel grid
        """
        idx = self.spatial_idx(x)
        self.radiance.scatter_add_(0,idx[...,None].expand_as(radiance),radiance)
        self.count.scatter_add_(0,idx,torch.ones_like(idx))
    
    def forward(self,x):
        """ query surface light field """
        idx = self.spatial_idx(x)
        radiance = self.radiance[idx]
        radiance[idx==-1] = 0 # if hit empty space, return zero radiance
        return {
            'rgb': radiance
        }


class TextureSLF(nn.Module):
    """ textured mesh based surface light field (unused) """
    def __init__(self,res,texture=None,Co=3):
        super(ExplicitSLF,self).__init__()
        self.res = res
        self.Co = Co
        if texture is None:
            texture = torch.randn(Co,res,res)*0.1 + 0.5
        self.register_parameter('feature',nn.Parameter(texture))
    
    def texture(self,uv):
        """ uv: Bx2"""
        uv = uv*2-1
        B,_ = uv.shape
        feat = NF.grid_sample(self.feature[None],uv.reshape(1,B,1,2),
                       mode='bilinear',align_corners=True).reshape(self.Co,B).T
        return feat
    
    def forward(self,uv,wo):
        feat = self.texture(uv)
        rgb = feat[...,:3]
        return {
            'rgb': rgb
        }