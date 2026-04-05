# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import io
import numpy as np
import os
import pathlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import torch
from torch import nn
import torch.nn.functional as F
import torch_interpolations

from iif.utils.image_io import show_image

folder_path = pathlib.Path(__file__).parent.resolve()

def parse_emor_file(inv=True):
    file_path = os.path.join(folder_path,  'invemor.txt') if inv else os.path.join(folder_path, 'emor.txt')
    with open(file_path, 'r') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]
    
    stride = 1 + 256
    names = []
    vectors = []
    for i in range(len(lines)//stride):
        name_line = lines[i*stride]
        name = name_line.split('=')[0].strip()
        names.append(name)
        lines_numbers = lines[i*stride+1: (i+1)*stride]
        numbers = [line.split() for line in lines_numbers]
        vector = np.float32(numbers).reshape(-1)
        vectors.append(vector)
    names = np.array(names)
    vectors = np.stack(vectors)
    return names, vectors

def parse_dorf_curves():
    with open(os.path.join(folder_path, 'dorfCurves.txt'), 'r') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]
    stride = 6
    names = []
    vectors = []
    for i in range(len(lines)//stride):
        line_sample = lines[i*stride: (i+1)*stride]
        n_i = '{}-{}-{}'.format(line_sample[0], line_sample[1], line_sample[2][0])
        n_b = '{}-{}-{}'.format(line_sample[0], line_sample[1], line_sample[4][0])
        names += [n_b]
        v_i = np.float32(line_sample[3].split())
        v_b = np.float32(line_sample[5].split())
        vectors += [v_b]
    names = np.array(names)
    vectors = np.stack(vectors)
    return names, vectors #(201, 1024)

def get_dorf_mean_basis(top=25):
    names, curves = parse_dorf_curves()
    mean = np.mean(curves, 0)
    curves = curves - mean[None]
    u, s, vh = np.linalg.svd(curves)
    scaled_basis = s[:top, None] * vh[:top]
    # scaled_basis = vh[:top]
    return mean, scaled_basis

def mono_increase_constraint(crf):
    diff = crf[1:] - crf[:-1]
    gap = -1 * np.min([0.0, diff.min()])
    diff += gap 
    diff /= diff.sum()
    crf = np.cumsum(diff)
    crf = np.concatenate([np.zeros((1)), crf])
    return crf


def mono_increase_constraint(crf):
    diff = crf[1:] - crf[:-1]
    diff_min = diff.min()
    gap = -diff_min if diff_min < 0 else 0 
    diff += gap 
    diff /= diff.sum()
    crf = torch.cumsum(diff, dim=0)
    crf = torch.cat([torch.zeros((1), device=crf.device), crf])
    return crf

class EmorCRF(nn.Module):
    def __init__(self, dim=3):
        super().__init__()
        self.dim = dim
        names, vectors = parse_emor_file(inv=False)
        self.register_buffer('f0', torch.FloatTensor(vectors[1])[None])
        self.register_buffer('basis', torch.FloatTensor(vectors[2:2+dim]))
        self.weight = nn.Parameter(torch.zeros(3, dim))
    
    def get_crf(self):
        crf = self.f0 + self.weight @ self.basis
        return crf
    
    def get_inv_crf(self):
        crf = self.get_crf()
        inv_crf = []
        for i in range(3):
            crf_ch = mono_increase_constraint(crf[i])
            x = torch.linspace(0, 1, len(crf_ch)).to(self.weight.device)
            interp_func = torch_interpolations.RegularGridInterpolator([crf_ch], x)
            inv_crf_ch = interp_func([x.contiguous()])
            inv_crf.append(inv_crf_ch)
        inv_crf = torch.stack(inv_crf, dim=0)
        return inv_crf

    def initialize_weight(self, crf):
        weight = self.cal_weight_fitting_crf(crf) #(3, dim)
        self.weight = nn.Parameter(torch.FloatTensor(weight).to(self.weight.device))

    def cal_weight_fitting_crf(self, crf):
        f0 = self.f0.detach().cpu().numpy()
        basis = self.basis.detach().cpu().numpy().T
        pseudo_inverse = np.linalg.inv(basis.T @ basis) @ basis.T
        weight = pseudo_inverse @ (crf - f0).T 
        return weight.T
    
    def forward(self, hdr, exposure):
        '''
        Input:
            hdr: (n, 3)
        Return:
            ldr: (n, 3)
        '''
        hdr = torch.clip(hdr*exposure, 0, 1)
        crf = self.get_crf()
        x = torch.linspace(0, 1, crf.size(1)).to(self.weight.device)
        ldr = []
        for i in range(3):
            hdr_ch = hdr[:, i]
            crf_ch = crf[i]
            interp_func = torch_interpolations.RegularGridInterpolator([x], crf_ch)
            ldr_ch = interp_func([hdr_ch.contiguous()])
            ldr.append(ldr_ch)
        ldr = torch.stack(ldr, dim=-1)
        return ldr
    
    def inverse(self, ldr, exposure):
        '''
        Input:
            ldr: (n, 3)
        Return:
            hdr: (n, 3)
        '''
        ldr = torch.clip(ldr, 0, 1)
        inv_crf = self.get_inv_crf()
        x = torch.linspace(0, 1, inv_crf.size(1)).to(self.weight.device)
        hdr = []
        for i in range(3):
            ldr_ch = ldr[:, i]
            inv_crf_ch = inv_crf[i]
            interp_func = torch_interpolations.RegularGridInterpolator([x], inv_crf_ch)
            hdr_ch = interp_func([ldr_ch.contiguous()])
            hdr.append(hdr_ch)
        hdr = torch.stack(hdr, dim=-1) / exposure
        return hdr

    def reg_weight(self):
        loss = torch.mean(self.weight ** 2)
        return loss
    
    def reg_monotonically_increasing(self):
        crf = self.get_crf() #(3, 1024)
        diff = crf[:, 1:] - crf[:, :-1] # should be all positive
        loss = torch.sum(F.relu(-diff))
        return loss
    
    def reg_smoothness(self):
        crf = self.get_crf()
        smoothness = crf[:, :-2] + crf[:, 2:] - 2 * crf[:, 1:-1]
        loss = torch.mean(smoothness ** 2)
        return loss
    
    def log_details(self, crf_gt=None):
        DPI = 150
        crf_pred = self.get_crf().detach().cpu().numpy()

        plt.ioff()
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(12, 4))
        x = np.linspace(0, 1, crf_pred.shape[1])

        ax0.title.set_text('CRF (R)')
        ax0.set_ylabel('Pixel intensity')
        ax0.set_xlabel('Irradiance')
        ax1.title.set_text('CRF (G)')
        ax1.set_xlabel('Irradiance')
        ax2.title.set_text('CRF (B)')
        ax2.set_xlabel('Irradiance')

        ax0.plot(x, crf_pred[0], c='r')
        ax1.plot(x, crf_pred[1], c='r')
        ax2.plot(x, crf_pred[2], c='r', label='pred')

        if crf_gt is not None:
            crf_gt = crf_gt.cpu().numpy()
            ax0.plot(x, crf_gt[0], c='b')
            ax1.plot(x, crf_gt[1], c='b')
            ax2.plot(x, crf_gt[2], c='b', label='GT')

        ax2.legend()
        
        # Save to Numpy array
        fig.canvas.draw()
        plot = torch.from_numpy(np.array(fig.canvas.buffer_rgba())).to(self.get_crf().device).permute(2,0,1).unsqueeze(0)[:, :3] / 255.0
        
        plt.close(fig)
        
        return {
            "plot": plot
        }
    
    def get_regularization_loss(self):
        return {
            "increasing": self.reg_monotonically_increasing(),
            "weight": self.reg_weight(),
        }