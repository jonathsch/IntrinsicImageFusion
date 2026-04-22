# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import einops
import torch
import torch.nn.functional as NF

import mitsuba
mitsuba.set_variant('cuda_ad_rgb')

from .ops import *

import drjit as dr

gettrace = getattr(sys, 'gettrace', None)
if gettrace is None:
    print('No sys.gettrace')
elif gettrace():
    dr.set_flag(dr.JitFlag.VCallRecord, False)
    dr.set_flag(dr.JitFlag.LoopRecord, False)


def ray_intersect(scene,xs,ds):  # Mitsuba 3.5
    """ warpper of mitsuba ray-mesh intersection 
    Args:
        xs: Bx3 pytorch ray origin
        ds: Bx3 pytorch ray direction
    Return:
        positions: Bx3 intersection location
        normals: Bx3 normals
        uvs: Bx2 uv coordinates
        idx: B triangle indices, -1 indicates no intersection
        valid: B whether a valid intersection
    """
    # convert pytorch tensor to mitsuba
    xs_mi = mitsuba.Point3f(*xs.T)
    ds_mi = mitsuba.Vector3f(*ds.T)
    rays_mi = mitsuba.Ray3f(xs_mi,ds_mi)
    
    ret = scene.ray_intersect_preliminary(rays_mi)
    idx = mitsuba.Int(ret.prim_index).torch().long()
    ret = ret.compute_surface_interaction(rays_mi)
    
    positions = ret.p.torch()
    normals = ret.n.torch()
    normals = NF.normalize(normals,dim=-1)
    
    # check if invalid intersection
    ts  = ret.t.torch()
    valid = (~ts.isinf())
    
    idx[~valid] = -1
    normals = double_sided(-ds,normals)
    return positions,normals,ret.uv.torch(),idx,valid


def ray_intersect_w_depth(scene,xs,ds):  # Mitsuba 3.5
    """ warpper of mitsuba ray-mesh intersection 
    Args:
        xs: Bx3 pytorch ray origin
        ds: Bx3 pytorch ray direction
    Return:
        positions: Bx3 intersection location
        normals: Bx3 normals
        uvs: Bx2 uv coordinates
        idx: B triangle indices, -1 indicates no intersection
        valid: B whether a valid intersection
    """
    # convert pytorch tensor to mitsuba
    xs_mi = mitsuba.Point3f(*xs.T)
    ds_mi = mitsuba.Vector3f(*ds.T)
    rays_mi = mitsuba.Ray3f(xs_mi,ds_mi)
    
    ret = scene.ray_intersect_preliminary(rays_mi)
    idx = mitsuba.Int(ret.prim_index).torch().long()
    ret = ret.compute_surface_interaction(rays_mi)
    
    positions = ret.p.torch()
    normals = ret.n.torch()
    normals = NF.normalize(normals,dim=-1)
    
    # check if invalid intersection
    ts  = ret.t.torch()
    valid = (~ts.isinf())
    ts = torch.nan_to_num(ts, nan=0.0, posinf=0.0, neginf=0.0)
    
    idx[~valid] = -1
    normals = double_sided(-ds,normals)
    return positions,normals,ret.uv.torch(),idx,valid, ts

# def ray_intersect(scene,xs,ds):  # Mitsuba 3.6
#     """ warpper of mitsuba ray-mesh intersection 
#     Args:
#         xs: Bx3 pytorch ray origin
#         ds: Bx3 pytorch ray direction
#     Return:
#         positions: Bx3 intersection location
#         normals: Bx3 normals
#         uvs: Bx2 uv coordinates
#         idx: B triangle indices, -1 indicates no intersection
#         valid: B whether a valid intersection
#     """
#     # convert pytorch tensor to mitsuba
#     xs_mi = mitsuba.Point3f(xs.T)
#     ds_mi = mitsuba.Vector3f(ds.T)
#     rays_mi = mitsuba.Ray3f(xs_mi,ds_mi)
    
#     ret = scene.ray_intersect_preliminary(rays_mi)
#     idx = mitsuba.Int(ret.prim_index).torch().long()
#     ret = ret.compute_surface_interaction(rays_mi)
    
#     positions = ret.p.torch().T
#     normals = ret.n.torch().T
#     normals = NF.normalize(normals,dim=-1)
    
#     # check if invalid intersection
#     ts  = ret.t.torch()
#     valid = (~ts.isinf())
    
#     idx[~valid] = -1
#     normals = double_sided(-ds,normals)
#     return positions,normals,ret.uv.torch().T,idx,valid

def path_tracing_det_diff(scene,emitter_net,material_net,
                          positions,wis,normals,uvs,triangle_idxs,
                          spp,indir_depth):
    """ Path trace diffuse shading with deterministic first intersection (from pixel center).
    Args:
        scene: mitsuba scene
        emitter_net: emitter object
        material_net: material object
        positions: Bx3 first intersection
        wis: Bx3 first intersection viewing direction
        normals: Bx3 first intersection normal
        uvs: Bx2 first intersection uvs
        triangle_idxs: B first intersection triangle indices
        spp: sampler per pixel
        indir_depth: path for indirect light
    Return:
        Lout: Bx3 diffuse shadings
    """
    device = positions.device
    emit_mask = (triangle_idxs!=-1) # drop invalid intersection
    Lout = torch.zeros_like(positions)
    
    if not emit_mask.any():
        return Lout
    
    position = positions[emit_mask]
    triangle_idx = triangle_idxs[emit_mask]
    mat = material_net(position=position, triangle_idx=triangle_idx) # get surface brdf
    # copy spp times 
    mat = {
        'albedo': mat['albedo'].repeat_interleave(spp,0),
        'roughness': mat['roughness'].repeat_interleave(spp,0),
        'metallic': mat['metallic'].repeat_interleave(spp,0)
    }
    normal = normals[emit_mask].repeat_interleave(spp,0)
    wo = -wis[emit_mask].repeat_interleave(spp,0)
    position = position.repeat_interleave(spp,0)
    
    
    B = emit_mask.sum()
    L = torch.zeros(B*spp,3,device=device)
    active_next = torch.ones(B*spp,dtype=bool,device=device)
    
    # importance sampling diffuse shading
    wi,brdf_pdf,brdf_weight = material_net.sample_diffuse(
        torch.rand(len(normal),2,device=device),normal)

    # next intersection
    position_next,normal,_,triangle_idx,vis = ray_intersect(scene,position+mitsuba.math.RayEpsilon*wi,wi)

    # get surface BRDF and Le
    mat_next = material_net(position_next)
    Le,emit_pdf,valid_next,_ = emitter_net.eval_emitter(position_next,wi,triangle_idx,mat_next['roughness'])
    
    # update throughput
    L[active_next] += brdf_weight*Le

    wo = -wi
    position = position_next
    
    # mask out unsued element
    active_next[active_next.clone()] = valid_next
    position = position[valid_next]
    triangle_idx = triangle_idx[valid_next]
    wo = wo[valid_next]
    normal = normal[valid_next]
    brdf_weight = brdf_weight[valid_next]
    
    # calculate indirect illumination
    with torch.no_grad():
        L_indir = trace_indirect(scene,emitter_net,material_net,position,triangle_idx,wo,normal,indir_depth,grad_depth=None)
    L[active_next] += brdf_weight*L_indir
    

    L = L.reshape(B,spp,3).mean(1)
    Lout[emit_mask] = L
    return Lout


def path_tracing_det_spec(scene,emitter_net,material_net,
                          roughness_level,
                          positions,wis,normals,uvs,triangle_idxs,
                          spp,indir_depth):
    """ Path trace specular shadings with deterministic first intersection (from pixel center).
    Args:
        scene: mitsuba scene
        emitter_net: emitter object
        material_net: material object
        roughness_level: roughness value for current specular shadings
        positions: Bx3 first intersection
        wis: Bx3 first intersection viewing direction
        normals: Bx3 first intersection normal
        uvs: Bx2 first intersection uvs
        triangle_idxs: B first intersection triangle indices
        spp: sampler per pixel
        indir_depth: path for indirect light
    Return:
        L0out: Bx3 specular shading Ls0
        L1out: Bx3 specular shading Ls1
    """
    device = positions.device
    emit_mask = (triangle_idxs != -1) # drop invalid intersection
    L0out = torch.zeros_like(positions)
    L1out = torch.zeros_like(positions)
    
    if not emit_mask.any():
        return L0out,L1out
    
    position = positions[emit_mask]
    triangle_idx = triangle_idxs[emit_mask]
    mat = material_net(position=position, triangle_idx=triangle_idx)
    # copy spp times
    mat = {
        'albedo': mat['albedo'].repeat_interleave(spp,0),
        'roughness': mat['roughness'].repeat_interleave(spp,0),
        'metallic': mat['metallic'].repeat_interleave(spp,0)
    }
    normal = normals[emit_mask].repeat_interleave(spp,0)
    wo = -wis[emit_mask].repeat_interleave(spp,0)
    position = position.repeat_interleave(spp,0)
    
    B = emit_mask.sum()
    L0 = torch.zeros(B*spp,3,device=device)
    L1 = torch.zeros(B*spp,3,device=device)
    active_next = torch.ones(B*spp,dtype=bool,device=device)
    
    # importance sampling brdf
    wi,_,brdf_weight0,brdf_weight1 = material_net.sample_specular(
        torch.rand(len(normal),2,device=device),wo,normal,roughness_level)

    # find next intersection
    position_next,normal,_,triangle_idx,vis = ray_intersect(scene,position+mitsuba.math.RayEpsilon*wi,wi)

    # get surface BRDF and Le
    mat_next = material_net(position=position_next, triangle_idx=triangle_idx)
    Le,_,valid_next, _ = emitter_net.eval_emitter(position_next,wi,triangle_idx,mat_next['roughness'])

    # update throughput
    L0[active_next] += brdf_weight0*Le
    L1[active_next] += brdf_weight1*Le
    
    wo = -wi
    position = position_next
    
    # mask out unsued element
    active_next[active_next.clone()] = valid_next
    position = position[valid_next]
    triangle_idx = triangle_idx[valid_next]
    wo = wo[valid_next]
    normal = normal[valid_next]
    brdf_weight0 = brdf_weight0[valid_next]
    brdf_weight1 = brdf_weight1[valid_next]
    
    
    # calculate indirect illumination
    with torch.no_grad():
        L_indir = trace_indirect(scene,emitter_net,material_net,position,triangle_idx,wo,normal,indir_depth, grad_depth=None)
    L0[active_next] += brdf_weight0*L_indir
    L1[active_next] += brdf_weight1*L_indir
    

    L0 = L0.reshape(B,spp,3).mean(1)
    L1 = L1.reshape(B,spp,3).mean(1)
    
    L0out[emit_mask] = L0
    L1out[emit_mask] = L1
    return L0out,L1out

def path_tracing(scene,
                 emitter_net,
                 material_net,
                 rays_o,
                 rays_d,
                 dx_du,
                 dy_dv,
                 spp,
                 depth,
                 grad_depth={'brdf': -1, 'emitter': -1}):
    """ Path trace current scene
    Args:
        scene: mitsuba scene
        emitter_net: emitter object
        material_net: material object
        rays_o: Bx3 ray origin
        rays_d: Bx3 ray direction
        dx_du,dy_dv: Bx3 ray differential
        spp: sampler per pixel
        depth: indirect illumination depth
    Return:
        L: Bx3 traced results
    """
    B = len(rays_o)
    device = rays_o.device

    g_depth = {
        'brdf': math.inf,
        'emitter': math.inf
    }
    g_depth.update(grad_depth or {})
    assert g_depth.get('brdf', 0) <= 1 or g_depth.get('emitter', 0) <= 1, "Propagating gradients from deeper is not implemented yet"
    
    # sample camera ray
    du,dv = torch.rand(2,len(rays_o),spp,1,device=device)-0.5
    wi = NF.normalize(rays_d[:,None]+dx_du[:,None]*du+dy_dv[:,None]*dv,dim=-1).reshape(-1,3)
    position = rays_o.repeat_interleave(spp,0)

    # compute first intersection
    position,normal,_,triangle_idx,vis = ray_intersect(scene,position,wi)

    # ===================== Bounce 0 =====================
    bounce = 0
    mat_context = torch.no_grad if g_depth['brdf'] < bounce else torch.enable_grad
    emitter_context = torch.no_grad if g_depth['emitter'] < bounce else torch.enable_grad

    with emitter_context():
        L,_,valid_next, _ = emitter_net.eval_emitter(position,wi,triangle_idx)

    valid_next = torch.ones_like(valid_next)  # TODO: Remove, just for debug
    
    # drop invalid intersection
    if not valid_next.any():
        return L
    position = position[valid_next]
    triangle_idx = triangle_idx[valid_next]
    normal = normal[valid_next]
    wo = -wi[valid_next]
    active_next = valid_next.clone()

    # obtain surface BRDF
    with mat_context():
        mat = material_net(position=position, triangle_idx=triangle_idx)

    # calculate direct illumination with MIS

    # sample emitter
    wi,emit_pdf,emit_triangle_idx = emitter_net.sample_emitter(
        torch.rand(len(position),device=device),
        torch.rand(len(position),2,device=device),
        position)
    
    # emit brdf
    with mat_context():
        emit_brdf,brdf_pdf = material_net.eval_brdf(wi,wo,normal,mat)
    
    # visibility test
    emit_position,emit_normal,_,triangle_idx,emit_valid = ray_intersect(scene,position+mitsuba.math.RayEpsilon*wi,wi)

    # Bounce 1 - Preliminary
    bounce = 1
    mat_context = torch.no_grad if g_depth['brdf'] < bounce else torch.enable_grad
    emitter_context = torch.no_grad if g_depth['emitter'] < bounce else torch.enable_grad

    emit_vis = (~emit_valid)|(emit_triangle_idx==triangle_idx)
    with emitter_context():
        emit_weight,_,_, _ = emitter_net.eval_emitter(emit_position,wi,triangle_idx)
    
    # goemetry term (assume double sided area light)
    G = (-wi*emit_normal).sum(-1).abs()\
      / (emit_position-position).pow(2).sum(-1).clamp_min(1e-6)
    G = torch.where(emit_valid,G,1).unsqueeze(-1) # env map use angular metric
    #G[G.isnan()] = 0.0
    emit_weight = emit_weight*emit_vis[...,None]*G/emit_pdf.clamp_min(1e-6)
    

    brdf_pdf = brdf_pdf * G
    w_mis = torch.where((emit_pdf>0)&(~brdf_pdf.isinf()),emit_pdf*emit_pdf/(emit_pdf*emit_pdf+brdf_pdf*brdf_pdf),0)
    w_mis[emit_pdf.isinf()|(brdf_pdf==0)] = 1
    L[active_next] += emit_brdf*emit_weight*w_mis

    # sample brdf
    with mat_context():
        wi,brdf_pdf,brdf_weight = material_net.sample_brdf(
            torch.rand(len(normal),device=device),
            torch.rand(len(normal),2,device=device),
            wo,normal,mat)
    
    # find next intersection
    position_next,normal,_,triangle_idx,vis = ray_intersect(scene,position+mitsuba.math.RayEpsilon*wi,wi)

    # ===================== Bounce 1 =====================
    bounce = 1
    mat_context = torch.no_grad if g_depth['brdf'] < bounce else torch.enable_grad
    emitter_context = torch.no_grad if g_depth['emitter'] < bounce else torch.enable_grad

    # If last bounce, evaluate the diffuse surface light field cache as well
    if depth > bounce:
        with mat_context():
            mat_next = material_net(position=position_next, triangle_idx=triangle_idx)
        
        # evaluate Le
        with emitter_context():
            Le,emit_pdf,valid_next, _ = emitter_net.eval_emitter(position_next,wi,triangle_idx,mat_next['roughness'])
    else:
        Le,emit_pdf,valid_next, _ = emitter_net.eval_emitter(position_next,wi,triangle_idx,roughness=-1)
    
    G = (-normal*wi).sum(-1).abs()\
      / (position-position_next).pow(2).sum(-1).clamp_min(1e-6)
    G = torch.where(valid_next,G,1)
    brdf_pdf = brdf_pdf * G[...,None]
    
    w_mis = torch.where((brdf_pdf>0)&(~emit_pdf.isinf()),brdf_pdf*brdf_pdf/(emit_pdf*emit_pdf+brdf_pdf*brdf_pdf),0)
    w_mis[brdf_pdf.isinf()|(emit_pdf==0)] = 1
    #w_mis[w_mis.isnan()] = 0
    L[active_next] += brdf_weight*Le*w_mis
    
    wo = -wi
    position = position_next
    
    # mask out unsued element
    position = position[valid_next]
    triangle_idx = triangle_idx[valid_next]
    wo = wo[valid_next]
    normal = normal[valid_next]
    brdf_weight = brdf_weight[valid_next]
    active_indirect = active_next.clone()
    active_indirect[active_next] = valid_next.clone()
    
    # disable gradient after first bounce
    # calculate indirect illumination
    with torch.no_grad():
        L_indir = trace_indirect(scene,emitter_net,material_net,position,triangle_idx,wo,normal,depth, grad_depth)
    L[active_indirect] += brdf_weight*L_indir

    L = L.reshape(B,spp,3).mean(1)
    return L

def path_tracing_single_obj_mat(scene,
                                emitter_net,
                                material_net,
                                segmentation_net,
                                rays_o,
                                rays_d,
                                dx_du,
                                dy_dv,
                                spp, 
                                albedo_transform):
    """ Path trace current scene
    Args:
        scene: mitsuba scene
        emitter_net: emitter object
        material_net: material object
        rays_o: Bx3 ray origin
        rays_d: Bx3 ray direction
        dx_du,dy_dv: Bx3 ray differential
        spp: sampler per pixel
    Return:
        L: Bx3 traced results
    """
    B = len(rays_o)
    device = rays_o.device
    trace_roughness = 0.0
    
    # sample camera ray
    du,dv = torch.rand(2,len(rays_o),spp,1,device=device)-0.5
    wi = NF.normalize(rays_d[:,None]+dx_du[:,None]*du+dy_dv[:,None]*dv,dim=-1).reshape(-1,3)
    position = rays_o.repeat_interleave(spp,0)
    
    # compute first intersection
    position,normal,_,triangle_idx,vis = ray_intersect(scene,position,wi)
    L,_,valid_next,mask = emitter_net.eval_emitter(position,wi,triangle_idx)
    
    # drop invalid intersection
    if not valid_next.any():
        return L
    position = position[valid_next]
    triangle_idx = triangle_idx[valid_next]
    normal = normal[valid_next]
    wo = -wi[valid_next]
    active_next = valid_next.clone()

    # obtain surface BRDF
    mat = material_net(position=position, triangle_idx=triangle_idx)
    segmentation = segmentation_net(position).argmax(-1).long()

    if albedo_transform is not None:
        albedo_transform = albedo_transform[segmentation]
        mat['albedo'] = einops.einsum(torch.cat([mat['albedo'], torch.ones_like(mat['albedo'][..., :1])], dim=-1), albedo_transform, "B D, B C D -> B C")

    # calculate direct illumination with MIS
    # sample emitter
    wi,emit_pdf,emit_triangle_idx = emitter_net.sample_emitter(
        torch.rand(len(position),device=device),
        torch.rand(len(position),2,device=device),
        position)
    
    # visibility test
    emit_position,emit_normal,_,triangle_idx,emit_valid = ray_intersect(scene,position+mitsuba.math.RayEpsilon*wi,wi)
    emit_vis = (~emit_valid)|(emit_triangle_idx==triangle_idx)
    emit_weight,_,_,_ = emitter_net.eval_emitter(emit_position,wi,triangle_idx)

    # goemetry term (assume double sided area light)
    G = (-wi*emit_normal).sum(-1).abs()\
      / (emit_position-position).pow(2).sum(-1).clamp_min(1e-6)
    G = torch.where(emit_valid,G,1).unsqueeze(-1) # env map use angular metric
    #G[G.isnan()] = 0.0
    emit_weight = emit_weight*emit_vis[...,None]*G/emit_pdf.clamp_min(1e-6)
    
    # emit brdf
    emit_brdf,brdf_pdf = material_net.eval_brdf(wi,wo,normal,mat)
    # brdf_pdf contains nan after 1st iter if "sample brdf" is enabled
    brdf_pdf = brdf_pdf * G
    w_mis = torch.where((emit_pdf>0)&(~brdf_pdf.isinf()),emit_pdf*emit_pdf/(emit_pdf*emit_pdf+brdf_pdf*brdf_pdf).clamp_min(1e-6),0)
    w_mis[emit_pdf.isinf()|(brdf_pdf==0)] = 1
    L[active_next] += emit_brdf*emit_weight*w_mis

    # sample brdf
    wi,brdf_pdf,brdf_weight = material_net.sample_brdf(
        torch.rand(len(normal),device=device),
        torch.rand(len(normal),2,device=device),
        wo,normal,mat)
    
    # find next intersection
    position_next,normal,_,triangle_idx,vis = ray_intersect(scene,position+mitsuba.math.RayEpsilon*wi,wi)
    mat_next = material_net(position=position_next, triangle_idx=triangle_idx)
    
    # evaluate Le
    Le,emit_pdf,valid_next,mask = emitter_net.eval_emitter(position_next,wi,triangle_idx,mat_next['roughness'], trace_roughness)
    G = (-normal*wi).sum(-1).abs()\
      / (position-position_next).pow(2).sum(-1).clamp_min(1e-6)
    G = torch.where(valid_next,G,1)
    brdf_pdf = brdf_pdf * G[...,None]
    
    w_mis = torch.where((brdf_pdf>0)&(~emit_pdf.isinf()),brdf_pdf*brdf_pdf/(emit_pdf*emit_pdf+brdf_pdf*brdf_pdf),0)
    w_mis[brdf_pdf.isinf()|(emit_pdf==0)] = 1
    #w_mis[w_mis.isnan()] = 0
    L[active_next] += brdf_weight*Le*w_mis

    L = L.reshape(B,spp,3).mean(1)
    return L


def path_tracing_single(scene,emitter_net,material_net,rays_o,rays_d,dx_du,dy_dv,spp):
    """ Path trace current scene
    Args:
        scene: mitsuba scene
        emitter_net: emitter object
        material_net: material object
        rays_o: Bx3 ray origin
        rays_d: Bx3 ray direction
        dx_du,dy_dv: Bx3 ray differential
        spp: sampler per pixel
    Return:
        L: Bx3 traced results
    """
    B = len(rays_o)
    device = rays_o.device
    trace_roughness = 0.0
    
    # sample camera ray
    du,dv = torch.rand(2,len(rays_o),spp,1,device=device)-0.5
    wi = NF.normalize(rays_d[:,None]+dx_du[:,None]*du+dy_dv[:,None]*dv,dim=-1).reshape(-1,3)
    position = rays_o.repeat_interleave(spp,0)
    
    # compute first intersection
    position,normal,_,triangle_idx,vis = ray_intersect(scene,position,wi)
    L,_,valid_next,mask = emitter_net.eval_emitter(position,wi,triangle_idx)
    
    # drop invalid intersection
    if not valid_next.any():
        return L
    position = position[valid_next]
    triangle_idx = triangle_idx[valid_next]
    normal = normal[valid_next]
    wo = -wi[valid_next]
    active_next = valid_next.clone()

    # obtain surface BRDF
    mat = material_net(position=position, triangle_idx=triangle_idx)

    # calculate direct illumination with MIS
    # sample emitter
    wi,emit_pdf,emit_triangle_idx = emitter_net.sample_emitter(
        torch.rand(len(position),device=device),
        torch.rand(len(position),2,device=device),
        position)
    
    # visibility test
    emit_position,emit_normal,_,triangle_idx,emit_valid = ray_intersect(scene,position+mitsuba.math.RayEpsilon*wi,wi)
    emit_vis = (~emit_valid)|(emit_triangle_idx==triangle_idx)
    emit_weight,_,_,_ = emitter_net.eval_emitter(emit_position,wi,triangle_idx)

    # goemetry term (assume double sided area light)
    G = (-wi*emit_normal).sum(-1).abs()\
      / (emit_position-position).pow(2).sum(-1).clamp_min(1e-6)
    G = torch.where(emit_valid,G,1).unsqueeze(-1) # env map use angular metric
    #G[G.isnan()] = 0.0
    emit_weight = emit_weight*emit_vis[...,None]*G/emit_pdf.clamp_min(1e-6)
    
    # emit brdf
    emit_brdf,brdf_pdf = material_net.eval_brdf(wi,wo,normal,mat)
    # brdf_pdf contains nan after 1st iter if "sample brdf" is enabled
    brdf_pdf = brdf_pdf * G
    w_mis = torch.where((emit_pdf>0)&(~brdf_pdf.isinf()),emit_pdf*emit_pdf/(emit_pdf*emit_pdf+brdf_pdf*brdf_pdf).clamp_min(1e-6),0)
    w_mis[emit_pdf.isinf()|(brdf_pdf==0)] = 1
    L[active_next] += emit_brdf*emit_weight*w_mis

    # sample brdf
    wi,brdf_pdf,brdf_weight = material_net.sample_brdf(
        torch.rand(len(normal),device=device),
        torch.rand(len(normal),2,device=device),
        wo,normal,mat)
    
    # find next intersection
    position_next,normal,_,triangle_idx,vis = ray_intersect(scene,position+mitsuba.math.RayEpsilon*wi,wi)
    mat_next = material_net(position=position_next, triangle_idx=triangle_idx)
    
    # evaluate Le
    Le,emit_pdf,valid_next,mask = emitter_net.eval_emitter(position_next,wi,triangle_idx,mat_next['roughness'], trace_roughness)
    G = (-normal*wi).sum(-1).abs()\
      / (position-position_next).pow(2).sum(-1).clamp_min(1e-6)
    G = torch.where(valid_next,G,1)
    brdf_pdf = brdf_pdf * G[...,None]
    
    w_mis = torch.where((brdf_pdf>0)&(~emit_pdf.isinf()),brdf_pdf*brdf_pdf/(emit_pdf*emit_pdf+brdf_pdf*brdf_pdf),0)
    w_mis[brdf_pdf.isinf()|(emit_pdf==0)] = 1
    #w_mis[w_mis.isnan()] = 0
    L[active_next] += brdf_weight*Le*w_mis

    L = L.reshape(B,spp,3).mean(1)
    return L

def trace_indirect(scene,emitter_net,material_net,position,triangle_idx,wo,normal,depth,grad_depth):
    """ trace indirect illumination
    Args:
        scene: mitsuba scene
        emitter_net: emitter object
        material_net: material object
        position: Bx3 current intersection location
        wo: Bx3 current viewing direction
        normal: Bx3 current normal
        indir_dpeth: indirect illumination depth
    Return:
        L: Bx3 indirect illumination
    """
    device = position.device
    B = position.shape[0]
    active_next = torch.ones(B,dtype=bool,device=device)# how many active rays
    throughput = torch.ones(B,3,device=device)
    L = torch.zeros(B,3,device=device)
    
    for current_depth in range(depth - 1):
        if not active_next.any():
            break
        # get material
        if current_depth == 0:
            mat = material_net(position=position, triangle_idx=triangle_idx)
        
        # sample emitter
        wi,emit_pdf,emit_triangle_idx = emitter_net.sample_emitter(
            torch.rand(len(position),device=device),
            torch.rand(len(position),2,device=device),
            position
        )

        # test visibility
        emit_position,emit_normal,_,triangle_idx,emit_valid = ray_intersect(scene,position+mitsuba.math.RayEpsilon*wi,wi)
        emit_vis = (~emit_valid)|(emit_triangle_idx==triangle_idx) # visible = not env (valid) + same triangle id
        emit_weight,_,_, _ = emitter_net.eval_emitter(emit_position,wi,triangle_idx)

        # goemetry term (assume double sided area light)
        G = (-wi*emit_normal).sum(-1).abs()\
          / (emit_position-position).pow(2).sum(-1).clamp_min(1e-12)
        G = torch.where(emit_valid,G,1).unsqueeze(-1) # env map use angular metric

        emit_weight = emit_weight*emit_vis[...,None]*G/emit_pdf.clamp_min(1e-12)
       
        # emit brdf
        emit_brdf,brdf_pdf = material_net.eval_brdf(wi,wo,normal,mat)
        brdf_pdf = brdf_pdf * G
        w_mis = torch.where((emit_pdf>0)&(~brdf_pdf.isinf()),emit_pdf*emit_pdf/(emit_pdf*emit_pdf+brdf_pdf*brdf_pdf),0)
        w_mis[emit_pdf.isinf()|(brdf_pdf==0)] = 1
        dL = throughput*emit_brdf*emit_weight*w_mis
        dL[dL.isnan()] = 0
        L[active_next] += dL
        
        
        # sample brdf
        wi,brdf_pdf,brdf_weight = material_net.sample_brdf(
            torch.rand(len(normal),device=device),
            torch.rand(len(normal),2,device=device),
            wo,normal,mat) 
        throughput = throughput*brdf_weight
        
        position_next,normal,_,triangle_idx,vis = ray_intersect(scene,position+mitsuba.math.RayEpsilon*wi,wi)
    
        mat_next = material_net(position=position_next, triangle_idx=triangle_idx)
        
        # evaluate Le
        Le,emit_pdf,valid_next, _ = emitter_net.eval_emitter(position_next,wi,triangle_idx,mat_next['roughness'])
        G = (-normal*wi).sum(-1).abs()\
          / (position-position_next).pow(2).sum(-1).clamp_min(1e-12)
        G = torch.where(valid_next,G,1)
        brdf_pdf = brdf_pdf * G[...,None]
        
        w_mis = torch.where((brdf_pdf>0)&(~emit_pdf.isinf()),brdf_pdf*brdf_pdf/(emit_pdf*emit_pdf+brdf_pdf*brdf_pdf),0)
        w_mis[brdf_pdf.isinf()|(emit_pdf==0)] = 1
        dL = throughput*Le*w_mis
        dL[dL.isnan()] = 0
        L[active_next] += dL
        
        wo = -wi
        position = position_next
        
        # mask out unsued element
        active_next[active_next.clone()] = valid_next
        position = position[valid_next]
        wo = wo[valid_next]
        normal = normal[valid_next]
        throughput = throughput[valid_next]
        mat = {
            'albedo': mat_next['albedo'][valid_next],
            'roughness': mat_next['roughness'][valid_next],
            'metallic': mat_next['metallic'][valid_next],
        }
    return L