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


def create_color_map(num_classes, seed=42):
    """Generate a color map for num_classes IDs."""
    g = torch.Generator().manual_seed(seed)  # deterministic colors
    colors = torch.randint(0, 256, (num_classes, 3), dtype=torch.uint8, generator=g) / 255
    if num_classes > 0:
        colors[0] = 0  # background is black
    return colors


class TriangleSegmentation(nn.Module):
    """ triangle emitters with diffuse radiance cache """
    def __init__(self, n_triangles=0, num_classes=0):
        """ 
        emitter_path: emitter parameter file
        slf_path: surface light field paramter file
        """
        super(TriangleSegmentation,self).__init__()
        self.module_logger = init_logger()
        
        # Define placeholder buffers
        self.register_buffer('num_classes', torch.tensor(num_classes, dtype=torch.int64))
        self.register_buffer('instance_id', torch.zeros(n_triangles, dtype=torch.int64))

        # Color map
        self.register_buffer('color_map', create_color_map(self.num_classes))

    def initialize(self, 
                   instance_id):
        self.instance_id = instance_id
        self.num_classes = torch.tensor(int(instance_id.max().item()+1), dtype=torch.int64)
        self.color_map = create_color_map(self.num_classes)
    
    def forward(self, triangle_idx, **kwargs):
        instance_id = self.instance_id[triangle_idx].long()
        return instance_id
    
    def forward_color(self, triangle_idx):
        instance_id = self.instance_id[triangle_idx]
        colors = self.color_map[instance_id]
        return colors
    
    def log_details(self, positions, directions, triangle_idx, b, h, w, spp, spp_batch):
        output = Batch()

        instances_hdr = batched_average(self, 
                                      Batch(triangle_idx=einops.rearrange(triangle_idx, "(b spp) ... -> b spp ...", spp=spp)),
                                      spp, spp_batch)
        instances_hdr = torch.round(instances_hdr).long()
        output['instances_hdr'] = einops.rearrange(instances_hdr, '(b h w) -> b 1 h w', b=b, h=h, w=w)

        instances = batched_average(self.forward_color, 
                                      Batch(triangle_idx=einops.rearrange(triangle_idx, "(b spp) ... -> b spp ...", spp=spp)),
                                      spp, spp_batch)
        output['instances'] = einops.rearrange(instances.clamp(0,1), '(b h w) c -> b c h w', b=b, h=h, w=w)
        return output
    

# class VertexSegmentation(nn.Module):
#     """ triangle emitters with diffuse radiance cache """
#     def __init__(self, n_vertices=0, num_segments=0):
#         """ 
#         emitter_path: emitter parameter file
#         slf_path: surface light field paramter file
#         """
#         super(VertexSegmentation,self).__init__()
#         self.module_logger = init_logger()
        
#         # Define placeholder buffers
#         self.register_buffer('num_segments', torch.tensor(num_segments, dtype=torch.int64))
#         self.register_buffer('instance_id', torch.zeros(n_vertices, dtype=torch.int64))
#         self.register_buffer('triangle_to_vertex', torch.zeros(n_vertices, 3, dtype=torch.int64))

#         # Color map
#         self.register_buffer('color_map', create_color_map(self.num_segments))

#     def initialize(self, 
#                    instance_id,
#                    triangle_to_vertex):
#         self.instance_id = instance_id
#         self.triangle_to_vertex = triangle_to_vertex
#         self.num_segments = torch.tensor(int(instance_id.max().item()+1), dtype=torch.int64)
#         self.color_map = create_color_map(self.num_segments)
    
#     def forward(self, triangle_idx):
#         instance_id = self.instance_id[triangle_idx]
#         return instance_id
    
#     def forward_color(self, triangle_idx):
#         instance_id = self.instance_id[triangle_idx]
#         colors = self.color_map[instance_id]
#         return colors
    
#     def log_details(self, positions, directions, triangle_idx, b, h, w, spp, spp_batch):
#         output = Batch()

#         instances = batched_average(self.forward_color, 
#                                       Batch(triangle_idx=einops.rearrange(triangle_idx, "(b spp) ... -> b spp ...", spp=spp)),
#                                       spp, spp_batch)

#         output['instances'] = einops.rearrange(instances.clamp(0,1), '(b h w) c -> b c h w', b=b, h=h, w=w)
#         return output
