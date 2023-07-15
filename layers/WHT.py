#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 12:14:21 2022

@author: Zephyr
"""

import numpy as np
import torch
from scipy.linalg import hadamard
from .common import find_min_power, SoftThresholding

def fwht(u, axis=-1, fast=False):
    """Multiply H_n @ u where H_n is the Hadamard matrix of dimension n x n.
    n must be a power of 2.
    Parameters:
        u: Tensor of shape (..., n)
        normalize: if True, divide the result by 2^{m/2} where m = log_2(n).
    Returns:
        product: Tensor of shape (..., n)
    """  
    if axis != -1:
        u = torch.transpose(u, -1, axis)
    
    n = u.shape[-1]
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'
    if fast:
        x = u[..., np.newaxis]
        for d in range(m)[::-1]:
            x = torch.cat((x[..., ::2, :] + x[..., 1::2, :], x[..., ::2, :] - x[..., 1::2, :]), dim=-1)
    else:
        H = torch.tensor(hadamard(n), dtype=torch.float, device=u.device)
        y = u @ H
    if axis != -1:
        y = torch.transpose(y, -1, axis)
    return y

def ifwht(u, axis=-1, fast=False):
    """Multiply H_n @ u where H_n is the Hadamard matrix of dimension n x n.
    n must be a power of 2.
    Parameters:
        u: Tensor of shape (..., n)
        normalize: if True, divide the result by 2^{m/2} where m = log_2(n).
    Returns:
        product: Tensor of shape (..., n)
    """  
    if axis != -1:
        u = torch.transpose(u, -1, axis)
    
    n = u.shape[-1]
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'
    if fast:
        x = u[..., np.newaxis]
        for d in range(m)[::-1]:
            x = torch.cat((x[..., ::2, :] + x[..., 1::2, :], x[..., ::2, :] - x[..., 1::2, :]), dim=-1)
        y = x.squeeze(-2) / n
    else:
        H = torch.tensor(hadamard(n), dtype=torch.float, device=u.device)
        y = u @ H /n
    if axis != -1:
        y = torch.transpose(y, -1, axis)
        
    return y

class WHTConv2D(torch.nn.Module):
    def __init__(self, height, width, in_channels, out_channels, pods = 1, residual=True):
        super().__init__()
        self.height = height       
        self.width = width
        self.height_pad = find_min_power(self.height)  
        self.width_pad = find_min_power(self.width)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pods = pods
        self.conv = torch.nn.ModuleList([torch.nn.Conv2d(in_channels, out_channels, 1, bias=False) for i in range(self.pods)])
        self.ST = torch.nn.ModuleList([SoftThresholding((self.height_pad, self.width_pad)) for i in range(self.pods)])
        self.v = torch.nn.ParameterList([torch.rand((self.height_pad, self.width_pad)) for i in range(self.pods)])
        self.residual = residual
        
        
    def forward(self, x):
        height, width = x.shape[-2:]
        if height!= self.height or width!=self.width:
            raise Exception('({}, {})!=({}, {})'.format(height, width, self.height, self.width))
     
        f0 = x
        if self.width_pad>self.width or self.height_pad>self.height:
            f0 = torch.nn.functional.pad(f0, (0, self.width_pad-self.width, 0, self.height_pad-self.height))
        
        f1 = fwht(f0, axis=-1)
        f2 = fwht(f1, axis=-2)
        
        
        f3 = [self.v[i]*f2 for i in range(self.pods)]
        f4 = [self.conv[i](f3[i]) for i in range(self.pods)]
        f5 = [self.ST[i](f4[i]) for i in range(self.pods)]
        
        f6 = torch.stack(f5, dim=-1).sum(dim=-1)
        
        f7 = ifwht(f6, axis=-1)
        f8 = ifwht(f7, axis=-2)
        
        y = f8[..., :self.height, :self.width]
        
        if self.residual:
            y = y + x
        return y
