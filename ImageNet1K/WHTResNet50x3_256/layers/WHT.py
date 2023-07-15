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

def hadamard_transform(u, axis=-1, fast=False):
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
        y = x.squeeze(-2) / 2**(m / 2)
    else:
        H = torch.tensor(hadamard(n), dtype=torch.float, device=u.device)
        y = u @ H.t()/np.sqrt(n)
    if axis != -1:
        y = torch.transpose(y, -1, axis)
        
    return y
    
class WHT1D(torch.nn.Module):
    """
    1D FWHT layer. We apply analysis along the last axis. 
    num_features: Length of the last axis, should be interger power of 2. If not, we pad 0s.
    residual: Apply shortcut connection or not
    retain_DC: Retain DC channel (the first channel) or not
    """
    def __init__(self, num_features, residual=True):
        super().__init__()
        self.num_features = num_features
        self.num_features_pad = find_min_power(self.num_features)  
        self.ST = SoftThresholding(self.num_features_pad)    
        self.v = torch.nn.Parameter(torch.rand(self.num_features_pad))
        self.residual = residual
        #with torch.no_grad():
        #    self.dense.weight.copy_(torch.eye(num_features))
         
    def forward(self, x):
        num_features = x.shape[-1]
        if num_features!= self.num_features:
            raise Exception('{}!={}'.format(num_features, self.num_features))
        if self.num_features_pad>num_features:
            f0 = torch.nn.functional.pad(x, (0, self.num_features_pad-num_features))
        else:
            f0 = x
        f1 = hadamard_transform(f0)

        f2 = self.v*f1
        f3 = self.ST(f2)
        f4 = hadamard_transform(f3)
        y = f4[..., :num_features]
        if self.residual:
            y = y + x
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
        
        f1 = hadamard_transform(f0, axis=-1)
        f2 = hadamard_transform(f1, axis=-2)
        
        f3 = [self.v[i]*f2 for i in range(self.pods)]
        f4 = [self.conv[i](f3[i]) for i in range(self.pods)]
        f5 = [self.ST[i](f4[i]) for i in range(self.pods)]
        
        f6 = torch.stack(f5, dim=-1).sum(dim=-1)
        
        f7 = hadamard_transform(f6, axis=-1)
        f8 = hadamard_transform(f7, axis=-2)
        
        y = f8[..., :self.height, :self.width]
        
        if self.residual:
            y = y + x
        return y

if __name__ == '__main__':
    x = torch.rand((10, 32, 128))
    model = WHT1D(x.shape[-1]) 
    y = model(x)
    
    x = torch.rand((10, 128, 32, 32))
    model2 = WHTConv2D(x.shape[-2], x.shape[-1], x.shape[-3], x.shape[-3]) 
    y = model2(x)