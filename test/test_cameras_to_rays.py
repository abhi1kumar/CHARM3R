"""
    Sample Run:
    python .py
"""
import os, sys

import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn as nn

np.set_printoptions   (precision= 2, suppress= True)
torch.set_printoptions(precision= 2, sci_mode= False)

from plot.common_operations import *
import plot.plotting_params as params
import matplotlib

from lib.helpers.homography_helper import get_intrinsics_from_fov
from lib.helpers.rays import compute_ndc_coordinates, cameras_to_rays, trans_to_crop_params

h = 512
w = h
scale = 0.5
shift_x, shift_y = -64, -64
# shift_x, shift_y = 0, 0

K = get_intrinsics_from_fov(h, w, fov_degree=64.56, four_by_four= True)



P2 = np.array([[ 398.25,    0.  ,  260.1 , -446.  ],
       [  -2.63,  400.9 ,  255.99,  158.49],
       [  -0.01,   -0.  ,    1.  ,   -1.72],
       [   0.  ,    0.  ,    0.  ,    1.  ]])


trans = np.array([[ scale, -0.  , shift_x],
       [ 0.  ,  scale, shift_y],
       [ 0.  , 0., 1.]])

trans_inv = np.array([[  1.13,  -0.  , -50.31],
       [  0.  ,   1.13, -21.06],
       [ 0.  , 0., 1.]])

print(trans)

vmax = 1.0
vmin = -vmax
cmap = 'seismic'
r = c = 2

device = torch.device('cpu')
xyd_grid = compute_ndc_coordinates(
    num_patches_x=w,
    num_patches_y=h,
    device= device).cpu().numpy() # H x W x 3

xyd_grid2 = compute_ndc_coordinates(
    crop_parameters= trans_to_crop_params(trans, K, im_width_height= (w, h), debug= True),
    num_patches_x=w,
    num_patches_y=h,
    device=device).cpu().numpy()  # H x W x 3


plt.figure(figsize= (12, 12), dpi= params.DPI)
plt.subplot(r,c,1)
plt.imshow(xyd_grid[:, :, 0], vmin= vmin, vmax= vmax, cmap= cmap)
plt.axis('off')
plt.subplot(r,c,3)
plt.imshow(xyd_grid[:, :, 1], vmin= vmin, vmax= vmax, cmap= cmap)
plt.axis('off')
plt.subplot(r,c,2)
plt.imshow(xyd_grid2[:, :, 0], vmin= vmin, vmax= vmax, cmap= cmap)
plt.axis('off')
plt.subplot(r,c,4)
plt.imshow(xyd_grid2[:, :, 1], vmin= vmin, vmax= vmax, cmap= cmap)
plt.axis('off')
plt.show()
plt.close()

cameras = torch.from_numpy(P2[np.newaxis, :]).type(torch.float32)
intrinsics = torch.from_numpy(K[:3, :3][np.newaxis, :]).type(torch.float32)
print(cameras)
print(intrinsics)
output = cameras_to_rays(
    cameras,
    intrinsics,
    return_mode='directions_plucker'
    ).cpu().numpy()

print(output.shape)