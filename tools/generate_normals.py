"""
    Sample Run:
    python test/compute_procthor_normal.py
"""
import os, sys
sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn as nn

from lib.helpers.file_io import read_numpy, save_numpy
from lib.helpers.homography_helper import get_intrinsics_from_fov

from plot.common_operations import convert_normal, savefig

from matplotlib import pyplot as plt
import glob
from kornia.geometry.depth import depth_to_normals
import warnings
import cv2


def get_my_normals(input_folder, save_normal_rgb= False):
    print("Input folder= {}".format(input_folder))
    depth_path_list = sorted(glob.glob(os.path.join(input_folder, "*.npy")))

    h = w = 512
    b = 1
    num_images = len(depth_path_list)
    output_folder = input_folder.replace("depth", "normal")
    os.makedirs(output_folder, exist_ok=True)

    for i, depth_path in enumerate(depth_path_list):
        a = read_numpy(depth_path)
        intrinsics = get_intrinsics_from_fov(w=w, h=h, fov_degree= 64.5615)

        depth = torch.zeros((b, 1, h, w))
        camera_matrix = torch.zeros((b, 3, 3))
        depth[0, 0] = torch.tensor(a)
        camera_matrix[0] = torch.tensor(intrinsics)
        depth = depth.cuda()
        camera_matrix = camera_matrix.cuda()

        # Normal computation
        # See https://kornia.readthedocs.io/en/latest/geometry.depth.html#kornia.geometry.depth.depth_to_normals
        normal = depth_to_normals(depth, camera_matrix)  # b x 3 x h x w
        normal = normal.permute(0, 2, 3, 1).cpu().detach().numpy()  # b x h x w x 3

        warnings.filterwarnings("ignore", message=".*kornia.*")

        # Save normals as numpy arrays
        basename = os.path.basename(depth_path)
        output_path = os.path.join(output_folder, basename)
        # normal_img = normal[0]
        # save_numpy(output_path, normal_img)

        # Save normals as colored images
        pred_norm_rgb = convert_normal(normal)  # (B, H, W, 3)
        pred_norm_rgb = pred_norm_rgb[0]  # (H, W, 3)
        output_path = os.path.join(output_path.replace(".npy", ".png"))
        cv2.imwrite(output_path, pred_norm_rgb)

        if (i + 1) % 5000 == 0 or (i + 1) == num_images:
            print("{} images done".format(i + 1))

list_of_heights = ['pitch0', 'height6', 'height12', 'height18', 'height24', 'height30']

for h in list_of_heights:
    if h == 'pitch0':
        input_folder    = os.path.join("data/carla/", h, "training" , "depth")
        get_my_normals(input_folder, save_normal_rgb= True)

    input_folder    = os.path.join("data/carla/", h, "validation" , "depth")
    get_my_normals(input_folder, save_normal_rgb= False)