import os
import zarr
import pickle
import tqdm
import numpy as np
import torch
import pytorch3d.ops as torch3d_ops
import torchvision
from termcolor import cprint
import re
import time


import numpy as np
import torch
import pytorch3d.ops as torch3d_ops
import torchvision
import socket
import pickle


def farthest_point_sampling(points, num_points=1024, use_cuda=True): #For test, increase the number of the sampled points. Original: 1014
    #points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        raise ValueError("输入点云为空数组！")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"输入需为 (N,3) 数组，但得到形状 {points.shape}")
    K = [num_points]
    if use_cuda:
        points = torch.from_numpy(points).cuda()
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0)
        sampled_points = sampled_points.cpu().numpy()
    else:
        points = torch.from_numpy(points)
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0)
        sampled_points = sampled_points.numpy()

    return sampled_points, indices

def preprocess_point_cloud(points, use_cuda=True):
    
    num_points = 1024

    # 正确写法（添加逗号）
    extrinsics_matrix = np.array([
        [-0.99999873, 0.00159265, 0, 0],
        [0.00159265, 0.99999873, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])

#box for our own point cloud
    # WORK_SPACE = [
    #     [-0.082, 0.10], 
    #     [-0.38, 0.065],
    #     [-1.075, -0.75]
    # ]
    #box for adroit's point cloud
    WORK_SPACE = [
    [-0.1, 0.27], 
    [-0.08, 0.4],
    [-0.075, 0.23]
]

    # scale
    point_xyz = points[..., :3]*0.0002500000118743628
    point_homogeneous = np.hstack((point_xyz, np.ones((point_xyz.shape[0], 1))))
    point_homogeneous = np.dot(point_homogeneous, extrinsics_matrix)
    point_xyz = point_homogeneous[..., :-1]
    points[..., :3] = point_xyz
    
     # crop
    points = points[np.where((points[..., 0] > WORK_SPACE[0][0]) & (points[..., 0] < WORK_SPACE[0][1]) &
                                 (points[..., 1] > WORK_SPACE[1][0]) & (points[..., 1] < WORK_SPACE[1][1]) &
                                 (points[..., 2] > WORK_SPACE[2][0]) & (points[..., 2] < WORK_SPACE[2][1]))]

    
    points_xyz = points[..., :3]
    points_xyz, sample_indices = farthest_point_sampling(points_xyz, num_points, use_cuda)
    sample_indices = sample_indices.cpu()
    points_rgb = points[sample_indices, 3:][0]
    points = np.hstack((points_xyz, points_rgb))
    return points

def boundary(WORK_SPACE):
    min_bound = np.array([WORK_SPACE[0][0], WORK_SPACE[1][0], WORK_SPACE[2][0]])  # 最小角点 [x_min, y_min, z_min]
    max_bound = np.array([WORK_SPACE[0][1], WORK_SPACE[1][1], WORK_SPACE[2][1]])  # 最大角点 [x_max, y_max, z_max]
    return min_bound, max_bound