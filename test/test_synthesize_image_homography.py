"""
    Sample Run:
    python .py
"""
import os, sys
sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn as nn

np.set_printoptions   (precision= 2, suppress= True)
torch.set_printoptions(precision= 2, sci_mode= False)

from plot.common_operations import *
import plot.plotting_params as params
import matplotlib

from lib.helpers.file_io import imread
from lib.helpers.homography_helper import get_intrinsics_from_fov
from lib.helpers.standard_camera_cpu import Standard_camera
import cv2

np.random.seed(0)

def get_cam2ground(cam_height, cam_pitch= 0.0):
    g2c = np.array([[1,                             0,                              0,          0],
                    [0, np.cos(np.pi / 2 + cam_pitch), -np.sin(np.pi / 2 + cam_pitch), cam_height],
                    [0, np.sin(np.pi / 2 + cam_pitch),  np.cos(np.pi / 2 + cam_pitch),          0],
                    [0,                             0,                              0,          1]])

    cam2ground = np.linalg.inv(g2c)
    return cam2ground


H = W = 512
r = 1
c = 3
src_intrinsics = get_intrinsics_from_fov(w= W, h= H, fov_degree= 64.5615)
src_cam2ground = get_cam2ground(cam_height= 1.76)
dst_intrinsics = src_intrinsics

height_arr = np.array([0, 6, 12, 18, 24, 30]) * 0.0254
height_arr = np.round(height_arr, 2)
new_height_arr = []
for i, h in enumerate(height_arr):
    new_height_arr.append(str(h) + "\n" + str(i))
height_folder_list = ["height" + str(x) for x in np.array([0, 6, 12, 18, 24, 30])]

src_image_folder  = "data/carla/height0/validation/image"
random_index_list = sorted(np.random.choice(np.arange(5000), 10, replace= False))

for iindex in random_index_list[:2]:
    key = str(iindex).zfill(6)
    src_image_path = os.path.join(src_image_folder, key + ".jpg")
    src_image = imread(src_image_path, rgb= True)

    for h, nh, hfolder in zip(height_arr, new_height_arr, height_folder_list):
        dst_cam2ground = get_cam2ground(cam_height= 1.76 + h)

        sc             = Standard_camera(dst_intrinsics, dst_cam2ground, (H, W), src_intrinsics, src_cam2ground, (H, W))
        trans_matrix   = sc.get_matrix(height=0)
        syn_image      = cv2.warpPerspective(src_image, trans_matrix, (H, W))

        gt_image_path  = os.path.join(src_image_folder.replace("height0", hfolder), key + ".jpg")
        gt_image       = imread(gt_image_path, rgb= True)

        plt.figure(figsize=(18,6), dpi= params.DPI)
        plt.subplot(r,c,1)
        plt.imshow(src_image)
        plt.axis('off')
        plt.title(key + ' Ht 0')
        plt.subplot(r,c,2)
        plt.imshow(syn_image)
        plt.axis('off')
        plt.title('Synthesized' + " Ht " + str(h))
        plt.subplot(r,c,3)
        plt.imshow(gt_image)
        plt.axis('off')
        plt.title('GT' + " Ht " + str(h))
        savefig(plt, "images/synthesize/synthesize_image_" + key + "_" + hfolder + ".png")
        # plt.show()
        plt.close()

    # break