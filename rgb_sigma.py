# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 12:30:06 2024

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
device="cpu "
import matplotlib.pyplot as plt

from run_nerf_helpers import *

from load_blender import load_blender_data
min_val = -1*pts_max
max_val = 1*pts_max
N=pts_res
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
lambda_sigma = 1e-3 # regularization for the lambda sigma (see the final loss in the loop)
lambda_rgb = 1e-3  # regularization for the lambda color (see the final loss in the loop)
N_rand = 1024# number of rays that are use during the training, IF YOU DO NOT HAVE ENOUGH RAM YOU CAN DECREASE IT BUT DO NOT NOT FORGET TO INCREASE THE N_iter!!!!
precrop_frac = 0.9 # do not change
start , N_iters = 0, 1000
N_samples = 20000 # numebr of samples along the ray
precrop_iters = 0
lrate = 5e-3 # learning rate
pts_res = 200 # point resolution of the pooint clooud        
pts_max = 3.725 # boundary of our point cloud
near = 2.
far = 6.
device = 'cpu'
# generate point cloud
pt_cloud, t_linspace, pt_cloud_indices = create_pointcloud(N = pts_res, min_val = -1*pts_max, max_val = pts_max)
pt_cloud = pt_cloud.to(device)
torch.manual_seed(42)
random_points = torch.randperm(pt_cloud.size(0))[:100]# Convert point_cloud_indices to a PyTorch tensor if it’s not already

# Get the selected random points and their grid indices as PyTorch tensors
selected_points_indices = pt_cloud_indices[random_points]
selected_points_indices=torch.tensor(selected_points_indices)

# Define the six neighboring offsets as a PyTorch tensor
neighbors_rel_coords = torch.tensor([
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1]
], dtype=torch.int32).to(selected_points_indices.device)  # Ensure compatibility with the device

# Calculate all neighboring coordinates for each selected point
all_neighbors_coords = selected_points_indices[:, None, :] + neighbors_rel_coords  # Shape: (100, 6, 3)


# Randomly select one neighbor for each point
num_selected_points = selected_points_indices.size(0)
torch.manual_seed(42)
random_neighbor_indices = torch.randint(0, 6, (num_selected_points,), device=selected_points_indices.device)  # Shape: (100,)
# Gather the randomly chosen neighbors
neighbor_coords = all_neighbors_coords[torch.arange(num_selected_points), random_neighbor_indices].unsqueeze(1)  # Shape: (100, 1, 3)
#define valid range
min_val=0
max_val=99
valid_mask=(neighbor_coords>=min_val) &( neighbor_coords<=max_val)
valid_mask = valid_mask.all(dim=-1).squeeze(-1)  # Shape: (100,)

# Remove points with invalid neighbors
valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)  # Indices of valid points
random_points = random_points[valid_indices]  # Filter random points
selected_points_indices = selected_points_indices[valid_indices]  # Filter selected point indices
neighbor_coords = neighbor_coords[valid_indices]  # 

pt_cloud_indices=torch.tensor(pt_cloud_indices)
# Expand dimensions to enable broadcasting
#neighbors_coords  (100,1,3)
pt_cloud_indices_expanded = pt_cloud_indices[None, :, :]  # Shape: (1, 1000000, 3)

# Compare each neighbor against all point cloud indices
matches = torch.all(neighbor_coords== pt_cloud_indices_expanded, dim=2)  # Shape: (100, 1000000)

# Get the indices where the equality holds
matching_indices = torch.nonzero(matches, as_tuple=False)  # Shape: (num_matches, 2)
# matching_indices[:, 1] -> Index in pt_cloud_indices
idx_pt_cloud= matching_indices[:, 1] 

neighbor_points=idx_pt_cloud
sigma_val = torch.ones(pt_cloud.size(0), 1).uniform_(0, 0.5).to(device)
rgb_val = torch.zeros(pt_cloud.size(0), 3).uniform_().to(device)

# Calculate L2 norms for RGB and sigma in a vectorized way
l2_rgb = torch.sum((rgb_val[random_points] - rgb_val[neighbor_points]) ** 2)
l2_sigma = torch.sum((sigma_val[random_points] - sigma_val[neighbor_points]) ** 2)

def LGS(masked_img, omega, lmbda):
    """
    Linear Gradient Solver (LGS) for inpainting.
    masked_img: masked image of size (M, N, 3)
    omega: binary mask of size (M, N)
    lmbda: regularization parameter
    """
    g = masked_img.copy()
    u = g.copy()
    m, n, _ = g.shape
    err = 1e-6
    max_iterations = 10

    for i in range(3):  # Iterate over channels
        u_channel = u[:, :, i]
        g_channel = g[:, :, i]

        for iteration in range(max_iterations):
            # Compute the gradient
            dE = (
                (2 * omega[1:-1, 1:-1] + 6) * u_channel[1:-1, 1:-1]
                - 2 * (
                    u_channel[2:, 1:-1] + u_channel[:-2, 1:-1]
                    + u_channel[1:-1, 2:] + u_channel[1:-1, :-2]
                    + omega[1:-1, 1:-1] * g_channel[1:-1, 1:-1]
                )
            )

            # Handle boundary gradients
            dE[0, :] = 0
            dE[-1, :] = 0
            dE[:, 0] = 0
            dE[:, -1] = 0

            # Solve the linear system: A @ u = b
            A = hessian_matrix(u_channel, omega, lmbda)
            b = A @ u_channel.flatten() - dE.flatten()

            # Use a sparse linear solver
            u_channel = sparse.linalg.cg(A, b, tol=err)[0].reshape((m, n))

            # Check for convergence
            if np.mean(np.abs(dE)) < err:
                break

        # Update the channel
        u[:, :, i] = u_channel

    return u
#pts_res = 200
 #delta = 7.45 / (pts_res - 1)
 #shift = -3.75
 #grid_points = ((pts - shift) / delta).round().long().clamp(0, pts_res - 1)

 # Convert (i, j, k) to a single linear index `q`
 #i_n, j_n, k_n = grid_points[:, 0], grid_points[:, 1], grid_points[:, 2]
# nn_index = i_n*pts_res + j_n * pts_res *pts_res+ k_n
# Example input
import numpy as np
import torch
pts = np.array([[1.2, 2.5, 3.06], [4.7, 5.1, 6.3]])
ptscloud= np.array([[1, 2, 3.76], [4.03, 5.14, 6.95],[14.7, 5.1, 5.3],[8.7, 1.1, 6.3],[4.7, 5.1, 6.3],[4.7, 5.1, 6.3]])  # Simulated large point cloud

# Find matching indices
nn_indices = find_nn(pts, ptscloud)
print(nn_indices)