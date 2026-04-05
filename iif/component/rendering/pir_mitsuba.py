# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hydra
from omegaconf import OmegaConf
import torch
import torch.nn.functional as NF

import drjit as dr
import mitsuba
mitsuba.set_variant('cuda_ad_rgb')

dr.set_flag(dr.JitFlag.VCallRecord, False)
dr.set_flag(dr.JitFlag.LoopRecord, False)

from iif.component.rendering.ops import double_sided
from iif.utils.model import freeze_model_params

import os
import sys
sys.path.append('..')


class PIRBSDF(mitsuba.BSDF):
    def __init__(self, props):
        mitsuba.BSDF.__init__(self, props)
        # default device for mitsuba
        device = torch.device(0)

        # Prepare BRDF model
        brdf_cfg = OmegaConf.load(props['brdf_cfg'])
        brdf_path = props['brdf_pt']
        self.brdf = hydra.utils.instantiate(brdf_cfg)
        self.brdf.load_state_dict(torch.load(brdf_path, weights_only=True))
        self.brdf.to(device)
        freeze_model_params(self.brdf)

        # Prepare the emitter model
        emitter_cfg = OmegaConf.load(props['emitter_cfg'])
        emitter_path = props['emitter_pt']
        self.emitter = hydra.utils.instantiate(emitter_cfg)
        self.emitter.load_state_dict(torch.load(emitter_path, weights_only=True))
        self.emitter.to(device)
        freeze_model_params(self.emitter)

        # specify flags
        reflection_flags   = mitsuba.BSDFFlags.SpatiallyVarying|mitsuba.BSDFFlags.DiffuseReflection|mitsuba.BSDFFlags.FrontSide | mitsuba.BSDFFlags.BackSide
        self.m_components  = [reflection_flags]
        self.m_flags = reflection_flags

    def sample(self, ctx, si, sample1, sample2, active):
        wi = si.to_world(si.wi).torch()
        normal = si.n.torch()
        position = si.p.torch()
        triangle_idx = mitsuba.Int(si.prim_index).torch().long()

        normal = double_sided(wi,normal)
        
        mat = self.brdf(position=position, triangle_idx=triangle_idx)

        # is_emitter = self.emitter.is_emitter[triangle_idx]

        wo,pdf,brdf_weight = self.brdf.sample_brdf(
            sample1.torch().reshape(-1),
            sample2.torch(),
            wi,normal,mat
        )
        # brdf_weight[is_emitter] = 1.0 # increase from 0 to 1 to fill the emitter region color
        
        pdf_mi = mitsuba.Float(pdf.squeeze(-1))
        wo_mi = mitsuba.Vector3f(wo)
        wo_mi = si.to_local(wo_mi)
        value_mi = mitsuba.Vector3f(brdf_weight)
        
        bs = mitsuba.BSDFSample3f()
        bs.pdf = pdf_mi
        bs.sampled_component = mitsuba.UInt32(0)
        bs.sampled_type = mitsuba.UInt32(+self.m_flags)
        bs.wo = wo_mi
        bs.eta = 1.0

        return (bs,value_mi)

    def eval(self, ctx, si, wo, active):
        wo = si.to_world(wo).torch()
        wi = si.to_world(si.wi).torch()
        triangle_idx = mitsuba.Int(si.prim_index).torch().long()

        ts  = si.t.torch()        
        normal = si.n.torch()
        position = si.p.torch()

        valid = (~ts.isinf())

        normal = double_sided(wi,normal)
        
        mat = self.brdf(position=position, triangle_idx=triangle_idx)
        
        # is_emitter = self.emitter.is_emitter[triangle_idx]
        
        brdf,_ = self.brdf.eval_brdf(wo,wi,normal,mat)

        brdf[~valid] = 0.
        # brdf[is_emitter[triangle_idx]]=0
        brdf = mitsuba.Vector3f(brdf)
        
        return brdf

    def pdf(self, ctx, si, wo,active):
        wo = si.to_world(wo).torch()
        wi = si.to_world(si.wi).torch()
        
        normal = si.n.torch()
        position = si.p.torch()

        normal = double_sided(wi,normal)
        
        mat = self.brdf(position=position, triangle_idx=triangle_idx)
        _,pdf = self.brdf.eval_brdf(wo,wi,normal,mat)
        pdf = mitsuba.Float(pdf.squeeze(-1))
        return pdf

    def eval_pdf(self, ctx, si, wo, active=True):
        wo = si.to_world(wo).torch()
        wi = si.to_world(si.wi).torch()
        triangle_idx = mitsuba.Int(si.prim_index).torch().long()
        
        normal = si.n.torch()
        position = si.p.torch()

        normal = double_sided(wi,normal)
        
        mat = self.brdf(position=position, triangle_idx=triangle_idx)

        # is_emitter = self.emitter.is_emitter[triangle_idx]
        
        brdf,pdf = self.brdf.eval_brdf(wo,wi,normal,mat)
        # brdf[is_emitter[triangle_idx]] = 0
        brdf = mitsuba.Vector3f(brdf)
        pdf = mitsuba.Float(pdf.squeeze(-1))
        
        return brdf,pdf
    
    def to_string(self,):
        return 'PIRBSDF'

# class PIRBSDF(mitsuba.BSDF):
#     def __init__(self, props):
#         mitsuba.BSDF.__init__(self, props)
#         # default device for mitsuba
#         device = torch.device(0)

#         # Prepare BRDF model
#         brdf_cfg = OmegaConf.load(props['brdf_cfg'])
#         brdf_path = props['brdf_pt']
#         self.brdf = hydra.utils.instantiate(brdf_cfg)
#         self.brdf.load_state_dict(torch.load(brdf_path, weights_only=True))
#         self.brdf.to(device)
#         freeze_model_params(self.brdf)

#         # Prepare the emitter model
#         emitter_cfg = OmegaConf.load(props['emitter_cfg'])
#         emitter_path = props['emitter_pt']
#         self.emitter = hydra.utils.instantiate(emitter_cfg)
#         self.emitter.load_state_dict(torch.load(emitter_path, weights_only=True))
#         self.emitter.to(device)
#         freeze_model_params(self.emitter)

#         # specify flags
#         reflection_flags   = mitsuba.BSDFFlags.SpatiallyVarying|mitsuba.BSDFFlags.DiffuseReflection|mitsuba.BSDFFlags.FrontSide | mitsuba.BSDFFlags.BackSide
#         self.m_components  = [reflection_flags]
#         self.m_flags = reflection_flags

#     @dr.syntax
#     def sample(self, ctx, si, sample1, sample2, active):
#         wi = si.to_world(si.wi).torch().T
#         normal = si.n.torch().T
#         position = si.p.torch().T
#         triangle_idx = mitsuba.Int(si.prim_index).torch().long()

#         normal = double_sided(wi,normal)
        
#         mat = self.brdf(position)

#         # is_emitter = self.emitter.is_emitter[triangle_idx]

#         wo,pdf,brdf_weight = self.brdf.sample_brdf(
#             sample1.torch().reshape(-1),
#             sample2.torch().T,
#             wi,normal,mat
#         )
#         # brdf_weight[is_emitter] = 1.0 # increase from 0 to 1 to fill the emitter region color
        
#         pdf_mi = mitsuba.Float(pdf.squeeze(-1))
#         wo_mi = mitsuba.Vector3f(wo.T)
#         wo_mi = si.to_local(wo_mi)
#         value_mi = mitsuba.Vector3f(brdf_weight.T)
        
#         bs = mitsuba.BSDFSample3f()
#         bs.pdf = pdf_mi
#         bs.sampled_component = mitsuba.UInt32(0)
#         bs.sampled_type = mitsuba.UInt32(+self.m_flags)
#         bs.wo = wo_mi
#         bs.eta = 1.0

#         return (bs,value_mi)

#     def eval(self, ctx, si, wo, active):
#         wo = si.to_world(wo).torch().T
#         wi = si.to_world(si.wi).torch().T
#         triangle_idx = mitsuba.Int(si.prim_index).torch().long()
        
#         normal = si.n.torch().T
#         position = si.p.torch().T

#         normal = double_sided(wi,normal)
        
#         mat = self.brdf(position)
        
#         # is_emitter = self.emitter.is_emitter[triangle_idx]
        
#         brdf,_ = self.brdf.eval_brdf(wo,wi,normal,mat)
#         # brdf[is_emitter[triangle_idx]]=0
#         brdf = mitsuba.Vector3f(brdf.T)
        
#         return brdf

#     def pdf(self, ctx, si, wo,active):
#         wo = si.to_world(wo).torch().T
#         wi = si.to_world(si.wi).torch().T
        
#         normal = si.n.torch().T
#         position = si.p.torch().T

#         normal = double_sided(wi,normal)
        
#         mat = self.brdf(position)
#         _,pdf = self.brdf.eval_brdf(wo,wi,normal,mat)
#         pdf = mitsuba.Float(pdf.squeeze(-1))
#         return pdf

#     def eval_pdf(self, ctx, si, wo, active=True):
#         wo = si.to_world(wo).torch().T
#         wi = si.to_world(si.wi).torch().T
#         triangle_idx = mitsuba.Int(si.prim_index).torch().long()
        
#         normal = si.n.torch().T
#         position = si.p.torch().T

#         normal = double_sided(wi,normal)
        
#         mat = self.brdf(position)

#         # is_emitter = self.emitter.is_emitter[triangle_idx]
        
#         brdf,pdf = self.brdf.eval_brdf(wo,wi,normal,mat)
#         # brdf[is_emitter[triangle_idx]] = 0
#         brdf = mitsuba.Vector3f(brdf.T)
#         pdf = mitsuba.Float(pdf.squeeze(-1))
        
#         return brdf,pdf
    
#     def to_string(self,):
#         return 'PIRBSDF'


mitsuba.register_bsdf("pir_bsdf", lambda props: PIRBSDF(props))