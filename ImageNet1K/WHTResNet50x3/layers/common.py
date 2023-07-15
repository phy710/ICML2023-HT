#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 12:13:40 2022

@author: Zephyr
"""
import torch

def find_min_power(x, p=2):
    y = 1
    while y<x:
        y *= p
    return y

class SoftThresholding(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.T = torch.nn.Parameter(torch.rand(self.num_features)/10)
              
    def forward(self, x):
        return torch.mul(torch.sign(x), torch.nn.functional.relu(torch.abs(x)-torch.abs(self.T)))
    