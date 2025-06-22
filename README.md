# 3D Reconstruction using Neural Radiance Fields (NeRF)

This project is part of Assignment 3 for the Computer Vision course (Fall 2023) at the University of Bern.

## 🧠 Project Overview

The goal of this project is to reconstruct a 3D scene and render it from arbitrary viewpoints using only a set of posed 2D images. This is achieved using a simplified version of Neural Radiance Fields (NeRF).

We represent the scene as a **voxel-based point cloud** consisting of over 8 million 3D points, each associated with learnable RGB and density (σ) values. These values are optimized during training to match the appearance of real images captured from known camera angles.

## 🏗️ Method Summary

- For each pixel in a training image, a ray is cast into the scene based on the camera pose.
- Points are sampled along each ray. Each sample is mapped to a voxel grid to efficiently find the nearest point in the point cloud.
- The color of each pixel is estimated using a weighted sum of sampled RGB values, where the weights depend on the density and transparency along the ray.
- This estimate is compared to the ground truth pixel color to compute the training loss.
- Over many iterations, the RGB and σ values in the voxel grid are updated to reconstruct the 3D scene.

## ⚙️ Regularization

To enforce smoothness in the 3D field:
- For a randomly selected set of points, one valid 6-connected neighbor is chosen.
- A regularization loss penalizes large differences in RGB and σ values between each point and its neighbor.

This helps prevent artifacts and encourages spatial consistency in the reconstructed scene.
Below you can find the result of rendering after 10000 iterations:
