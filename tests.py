# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 17:04:03 2024

@author: Bazghandi
"""

import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm, trange

import matplotlib.pyplot as plt
def create_pointcloud(N, min_val, max_val):
    '''
    N - number of points along the axis
    min_vaL and max_val: points are sampled between these two points
    '''

    t = np.linspace(min_val, max_val, N)
    t_index = np.arange(N)
    query_pts = np.stack(np.meshgrid(t, t, t), -1).astype(np.float32)
    query_indices = np.stack(np.meshgrid(t_index, t_index, t_index), -1).astype(np.int16)
    flat = query_pts.reshape([-1, 3])
    flat_indices = query_indices.reshape([-1, 3])
   
    return torch.tensor(flat), t, flat_indices

pts_max = 3.725 
pt_cloud, t_linspace, pt_cloud_indices = create_pointcloud(N = 5, min_val = -1*pts_max, max_val = pts_max)


test_pts = pt_cloud[[0, 62, 124]]  # Select points at specific indices
def find_nn(pts, ptscloud):
    """
    pts: points along the ray of size (M,3)
    ptscloud: points in the pointcloud (KX3), where K=200X200X200


    :returns nn_index: the nearest index for every point in pts 
    """
    pts_res = 5
    delta = 7.45 / (pts_res - 1)
    shift = -3.75
    grid_points = ((pts - shift) / delta).round().long().clamp(0, pts_res - 1)

    # Convert (i, j, k) to a single linear index `q`
    i_n, j_n, k_n = grid_points[:, 0], grid_points[:, 1], grid_points[:, 2]
    nn_index = i_n*pts_res + j_n * pts_res *pts_res+ k_n
    return nn_index.long()

print("Test Points:", test_pts)
test_indices = find_nn(test_pts, pt_cloud)
print(test_indices.numpy())