'''
READ THIS

you should run thhis python code from Command Prompt.

To install pytorch-cpu: In anaconda run the following :conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 cpuonly -c pytorch

if you have gpu you can run also your code in GPU:
you can isntall pytorch-gpu (based on cuda version)
For example 

# CUDA 11.6
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
# CUDA 11.7
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia

For more information: 
https://pytorch.org/get-started/previous-versions/


There are bunch of video on yourtuve that show how to install pytorch.

'''


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


def raw2outputs(raw, ray_steps, rays_d,N_rand,N_samples):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: Predicted rgb and density of size (num_rays, num_samples along ray, 4).
        z_vals: Integration time of size (num_rays, num_samples along ray).
        rays_d:Direction of each ray of size (num_rays, 3)
    Returns:
        rgb_map: Estimated RGB color of a ray of size (num_rays, 3.)
    """

 
    '''
    TODO
    Implement volume rendering
    '''
    '''
    by weights we mean the all coefficients the multiplies the color \hat C(r) = \sum_{n=1}^{N} T_n (1-\exp(-\sigma_n\delta_n))c_n
    weights_n = T_n (1-\exp(-\sigma_n\delta_n))
    '''
    # Define function to convert raw densities to alpha values
    raw2alpha = lambda raw, delta, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * delta)
    z_vals=ray_steps.numpy()
     # expand this for all rays
    z_vals = np.tile(z_vals, (N_rand , 1))
    z_vals = torch.tensor(z_vals)
    # Compute distances between consecutive samples
    delta = z_vals[..., 1:] - z_vals[..., :-1]  # [num_rays, num_samples - 1]
    delta = torch.cat([delta, torch.Tensor([1e10]).expand(delta[..., :1].shape)], -1)  # [num_rays, num_samples]
    delta = delta * torch.norm(rays_d[..., None, :], dim=-1)  # Scale by ray length

    # Normalize RGB values to the range [0, 1]
    rgb = torch.sigmoid(raw[..., :3])  # [num_rays, num_samples, 3]

    # Optional noise addition to density for regularization
    raw_noise_std=0
    noise=0
    white_bkgd=False
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std  # Gaussian noise
        noise = noise.to(raw.device)  # Ensure noise is on the same device as raw

    # Compute alpha values using raw2alpha
    density_scale_factor=1
    alpha = raw2alpha(raw[..., 3] * density_scale_factor + noise, delta)  # [num_rays, num_samples]

    # Compute weights for color accumulation
    weights = alpha * torch.cumprod(
    torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device), 1. - alpha + 1e-10], -1), -1
)[:, :-1] # [num_rays, num_samples]

    # Accumulate color along the ray using weights
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # [num_rays, 3]

    # Apply white background if specified
    if white_bkgd:
        acc_map = torch.sum(weights, dim=-1)  # Accumulated opacity
        rgb_map = rgb_map + (1. - acc_map[..., None])
        rgb_map=rgb_map
       

    return rgb_map

    
def find_nn(pts, ptscloud):
    """
    pts: points along the ray of size (M,3)
    ptscloud: points in the pointcloud (KX3), where K=200X200X200


    :returns nn_index: the nearest index for every point in pts 
    """
    # Ensure pts and ptscloud are tensors
    Δ = 7.45 / 199
    grid_size = 200
    shift = 3.75
    # Map each point to grid indices
    indices = torch.round((pts + shift) / Δ).to(dtype=torch.int)
    indices = torch.clamp(indices, 0, grid_size - 1)

    # Convert (i,j,k) to flat index q
    i, j, k = indices[:, 0], indices[:, 1], indices[:, 2]
    nn_index = i * grid_size + j * grid_size**2 + k
    
    return  nn_index

def render_rays_discrete(ray_steps, rays_o, rays_d,N_rand, N_samples,pts_max,pt_cloud,pt_cloud_indices, rgb_val, sigma_val):
    """
    ray_steps: the scalar list of size (N_samples) of steps (see TODO below)
    rays_o: origin of the rays of size (NX3) 
    rays_d: direction of the rays of size (NX3)
    N_samples: number of samples along the ray
    pt_cloud: point cloud of size (KX3)
    rgb_val: rgb values for every point in the  point cloud of size (KX3)
    sigma_val: density values for every point in the point cloud of size (KX1)

    N --> number of rays
    K --> total number of points in the point cloud
    
    TODO: 
    Inside this function:
    1. generate points along the ray via: pts = ray_o + ray_d*ray_steps --> shape (num_rays, number samples along ray, 3)
    2. Find the nearest indices/points for each point along the ray.
    3. render rgb values for rays using the raw2outputs() function

    
    :returns rgb_val_rays: rgb values for the rays of size (NX3)
    """
    #first lets reshape the ray_o,ray_d and ray_steps 
   
    # Unsqueezing along the second dimension to get shape (N_rand, 1, 3)
    rays_o=rays_o.unsqueeze(1)
    # Repeat along the second dimension to get shape (N_rand, N_samples, 3)
    rays_o = rays_o.expand(-1, N_samples, -1)
    rays_d_original_shape = rays_d.clone()
    rays_d=rays_d.unsqueeze(1)
    ray_steps_original_shape=ray_steps.clone()
    ray_steps=ray_steps.unsqueeze(0).unsqueeze(-1)
    pts = rays_o + rays_d*ray_steps
    pts=pts.reshape(-1,3)
    raw = torch.zeros(N_rand * N_samples,4)
    nn_index = find_nn(pts, pt_cloud)
   
    raw[:, 0] = rgb_val[nn_index, 0]  # Assigning only the first channel for simplicity
    raw[:, 1] = rgb_val[nn_index, 1] # Assign second channel if needed
    raw[:, 2] = rgb_val[nn_index, 2] # Assign third channel if needed
    raw[:, 3] = sigma_val[nn_index].squeeze()
        
    
    raw=raw.view(N_rand,N_samples,4)   
    rays_d= rays_d_original_shape
    ray_steps= ray_steps_original_shape
    rgb_val_rays=raw2outputs(raw,ray_steps, rays_d,N_rand,N_samples)
    
    return rgb_val_rays

def regularize_rgb_sigma(point_cloud,point_cloud_indices, rgb_values, sigma_values):
    """
    point_cloud: your point cloud of size (KX3).
    rgb_values: rgb values of the points in the point cloud of size (KX3).
    sigma_values:  Sigma values of the points in the point cloud of size (KX1).
    
    K --> The total number of points in your point cloud (200X200X200)
    TODO: 
    Inside this function:
    Implement  regularization terms for rgb and sigma

    :returns 
    l2_rgb: regularization for rgb - scalar
    l2_sigma: regularization for density - scalar 
    """
    torch.manual_seed(42)
    K = point_cloud.shape[0]
    device = point_cloud.device
    grid_size = 200

    # 1. Select random points
    num_points = 100
    # 🔧 Ensure tensor format
    if isinstance(point_cloud_indices, np.ndarray):
        point_cloud_indices = torch.tensor(point_cloud_indices, dtype=torch.long, device=device)

    rand_ids = torch.randperm(K)[:num_points].to(device)
    base_indices = point_cloud_indices[rand_ids].to(device).long()  # make sure it's tensor
    # 2. Define all 6 neighbor directions
    neighbor_offsets = torch.tensor([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ], device=device, dtype=torch.long)

    chosen_neighbors = []
    valid_centers = []

    for i in range(base_indices.shape[0]):
        point = base_indices[i]
        # Shuffle directions to randomize neighbor selection order
        shuffled = neighbor_offsets[torch.randperm(6)]
        for offset in shuffled:
            neighbor = point + offset
            if ((neighbor >= 0) & (neighbor < grid_size)).all():
                chosen_neighbors.append(neighbor)
                valid_centers.append(point)
                break  # found one valid neighbor, stop

    if len(valid_centers) == 0:
        return torch.tensor(0.0), torch.tensor(0.0)  # fallback if all are invalid

    center_indices = torch.stack(valid_centers)
    neighbor_indices = torch.stack(chosen_neighbors)

    # Convert to flat indices
    def to_flat_index(indices):
        return indices[:, 0] * grid_size + indices[:, 1] * grid_size**2 + indices[:, 2]

    center_flat = to_flat_index(center_indices)
    neighbor_flat = to_flat_index(neighbor_indices)

    # Compute regularization losses
    rgb_diff = rgb_values[center_flat] - rgb_values[neighbor_flat]
    sigma_diff = sigma_values[center_flat] - sigma_values[neighbor_flat]

    l2_rgb = torch.sum(rgb_diff ** 2)
    l2_sigma = torch.sum(sigma_diff ** 2)
    return l2_rgb, l2_sigma


def train():
    K = None
    device = 'cpu'

    '''
    Do not change below parameters !!!!!!!!
    '''
    N_rand = 1024# number of rays that are use during the training, IF YOU DO NOT HAVE ENOUGH RAM YOU CAN DECREASE IT BUT DO NOT NOT FORGET TO INCREASE THE N_iter!!!!
    precrop_frac = 0.9 # do not change
    start , N_iters = 0, 10000
    N_samples = 200 # numebr of samples along the ray
    precrop_iters = 0
    lrate = 5e-3 # learning rate
    pts_res = 200 # point resolution of the pooint clooud        
    pts_max = 3.725 # boundary of our point cloud
    near = 2.
    far = 6.

    # You can play with this hyperparameters
    lambda_sigma = 1e-3 # regularization for the lambda sigma (see the final loss in the loop)
    lambda_rgb = 1e-3  # regularization for the lambda color (see the final loss in the loop)


    main_folder_name = 'Train_lego' # folder name where the output images, out variables will be estimated 
    # load dataset
    images, poses, render_poses, hwf, i_split = load_blender_data('data/nerf_synthetic/lego', True, 8)
    print('Loaded blender', images.shape, render_poses.shape, hwf)
    i_train, _, _ = i_split
    print('\n i_train: ', i_train)
    # get white background
    images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])


    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])


    # generate point cloud
    pt_cloud, t_linspace, pt_cloud_indices = create_pointcloud(N = pts_res, min_val = -1*pts_max, max_val = pts_max)
    pt_cloud = pt_cloud.to(device)
    save_folder_test = os.path.join('logs', main_folder_name) 
    os.makedirs(save_folder_test, exist_ok=True)
    torch.save(pt_cloud.cpu(), os.path.join(save_folder_test, 'pts_clous.tns'))
    torch.save(torch.tensor(t_linspace), os.path.join(save_folder_test, 't_linspace.tns'))
    torch.save(torch.tensor(pt_cloud_indices).long(), os.path.join(save_folder_test, 'pt_cloud_indices.tns'))
    sigma_val = torch.ones(pt_cloud.size(0), 1).uniform_(0, 0.5).to(device)
    rgb_val = torch.zeros(pt_cloud.size(0), 3).uniform_().to(device)

    # do not make any change
    sigma_val.requires_grad = True
    rgb_val.requires_grad = True

    optimizer = torch.optim.Adam([{'params':sigma_val},
                                 {'params':rgb_val}],
                                 lr=lrate,
                                 betas=(0.9, 0.999))


    for i in trange(start, N_iters):
        # Random from one image
        img_i = np.random.choice(i_train)
        target = images[img_i]
        target = torch.Tensor(target).to(device)
        pose = poses[img_i, :3,:4]

        if N_rand is not None:
            # for every pixel in the image, get the rray origin and ray direction
            rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)
            
            if  i < precrop_iters:
                '''
                if this is True, at the  beggining it will sample rays only from the 'center' of the image to avaid bad local minima
                '''
                dH = int(H//2 * precrop_frac)
                dW = int(W//2 * precrop_frac)
                coords = torch.stack(
                    torch.meshgrid(
                        torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                        torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                    ), -1)
                if i == start:
                    print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {precrop_iters}")                
            else:
                coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

            coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
            select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
            select_coords = coords[select_inds].long()  # (N_rand, 2)
            # select final ray_o and ray_d
            rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

            t_vals = torch.linspace(0., 1., steps=N_samples)
            z_vals = near * (1.-t_vals) + far * (t_vals)
           

            rgb_map = render_rays_discrete(ray_steps = z_vals,
                                           rays_o = rays_o,
                                           rays_d = rays_d,
                                           N_rand=N_rand,
                                           N_samples = N_samples,
                                           pts_max=pts_max,
                                           pt_cloud = pt_cloud,
                                           pt_cloud_indices=pt_cloud_indices,
                                           rgb_val = rgb_val,
                                           sigma_val = sigma_val) # CHANGE THIS
            # Note that the rgb_map MUST have the same shape as the target_s !!!!
         
            # do not make any change          
            optimizer.zero_grad()
            img_loss = img2mse(rgb_map, target_s)
            reg_loss_rgb, reg_loss_sigma = regularize_rgb_sigma(point_cloud =pt_cloud,point_cloud_indices=pt_cloud_indices, rgb_values=rgb_val, sigma_values = sigma_val) # DO NOT FORGET TO CHANGE THIS
            loss = img_loss + lambda_rgb*reg_loss_rgb + lambda_sigma*reg_loss_sigma # --> this is the loss we minimize
            psnr = mse2psnr(img_loss)
            loss.backward()
            optimizer.step()
            
          
        if i%1000==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()} loss image: {img_loss.item()}")
            print("Rendered RGB values (first 5):", rgb_map[:5])
            print("Target RGB values (first 5):", target_s[:5])
           
            
            # Assuming `rgb_map` is the current output
            debug_rgb_map = rgb_map.detach().cpu().numpy()  # Convert to NumPy
            debug_rgb_map = rearrange(debug_rgb_map, '(w h) d -> w h d', w=32)  # Rearrange dimensions
            debug_rgb_map_8bit = to8b(debug_rgb_map)  # Normalize to 8-bit (0-255)
            
            # Save the debug image
            debug_folder = os.path.join('logs', 'debug_images')
            os.makedirs(debug_folder, exist_ok=True)
            debug_filename = os.path.join(debug_folder, f'iter_{i:05d}.png')
            imageio.imwrite(debug_filename, debug_rgb_map_8bit)

        if i%1000==0: # 
            '''
            YOU DONO NOT NEED TO MAKE ANY CHANGE HERE EXCEPT  render_rays_discrete FUNCTION !!!!!
            at 1000-th iteraion the bulldozer should be appeared when trained with the default hyperparameters
            We save some intermediate images.
            The first 100 images are from the training set and the rest are novel views!. To speed up the generation we renderer every 8th pose/image
            '''
            save_folder_test_img = os.path.join('logs', main_folder_name, f"{i:05d}") 
            os.makedirs(save_folder_test_img, exist_ok=True)
            torch.save(rgb_val.detach().cpu(), os.path.join(save_folder_test_img, 'rgb_{:03d}.tns'.format(i)))
            torch.save(sigma_val.detach().cpu(), os.path.join(save_folder_test_img, 'sigma_{:03d}.tns'.format(i)))
            print(poses.shape[0])
            for j in trange(0,poses.shape[0], 8):
                
                pose = poses[j, :3,:4]
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                

                chunk = 200
                # N_rand = chunk
                rgb_image = []
                for k in range(int(coords.size(0)/chunk)):
                    select_coords = coords[k*chunk: (k+1)*chunk].long()  # (N_rand, 2)
                    rays_o_batch = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                    rays_d_batch = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

                    
                    t_vals = torch.linspace(0., 1., steps=N_samples)
                    z_vals = near * (1.-t_vals) + far * (t_vals)

                    with torch.no_grad():
                        rgb_map = render_rays_discrete(ray_steps = z_vals,
                                rays_o = rays_o_batch,
                                rays_d = rays_d_batch,
                                N_rand=chunk,
                                pts_max=pts_max,
                                N_samples = N_samples,
                                pt_cloud = pt_cloud,
                                pt_cloud_indices=pt_cloud_indices,
                                rgb_val = rgb_val,
                                sigma_val = sigma_val) # CHANGE THIS
                        
                    rgb_image.append(rgb_map)
                    
                rgb_image = torch.cat(rgb_image)
               
                
                rgb_image = rearrange(rgb_image, '(w h) d -> w h d', w = W)
                

                rgbimage8 = to8b(rgb_image.cpu().numpy())
                filename = os.path.join(save_folder_test_img, '{:03d}.png'.format(j))
                imageio.imwrite(filename, rgbimage8)

if __name__=='__main__':
    # torch.set_default_tensor_type('torch.cuda.FloatTensor') # UNCOMMENT THIS IF YOU NEED TO RUN IT IN GPU

    train()