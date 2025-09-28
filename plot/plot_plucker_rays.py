"""
    Sample Run:
    python .py
"""
import os, sys
sys.path.append(os.getcwd())

import os.path as osp
import glob
import numpy as np
import torch
import torch.nn as nn

np.set_printoptions   (precision= 2, suppress= True)
torch.set_printoptions(precision= 2, sci_mode= False)

from plot.common_operations import *
import plot.plotting_params as params
import matplotlib
from lib.helpers.rays import cameras_to_rays, trans_to_crop_params

BATCH_SIZE = 4
height_array = np.arange(4)*0.15

calibs = torch.eye(4).cuda().unsqueeze(0).repeat(BATCH_SIZE, 1, 1) # B x 4 x 4

# compute camera matrices, intrinsics and finally rays
ones    = torch.zeros_like(calibs[:, 0].unsqueeze(1))
ones[:, :, 3] = 1.0

ref_ext = torch.eye(4).type(calibs.dtype).to(calibs.device).unsqueeze(0).repeat(BATCH_SIZE, 1, 1) # B x 4 x 4
ref_ext[:, 1, 3] = 1.519076                                            # B x 4 x 4
intrinsics = calibs[:, :3, :3].clone()
intrinsics[: ,  0, 0] = 405.2535049749
intrinsics[: ,  1, 1] = 405.2535049749
intrinsics[: ,  0, 2] = 256.0
intrinsics[: ,  1, 0] = 256.0

intrinsics_4 = torch.zeros_like(ref_ext)
intrinsics_4[:, :3, :3] = intrinsics
intrinsics_4[:,  3,  3] = 1.0

gd_to_cam = calibs.clone()                                             # B x 4 x 4
for b, ht in zip(range(BATCH_SIZE), height_array):
    gd_to_cam[b, 1, 3] = 1.519076 + ht

cameras   = intrinsics_4 @ gd_to_cam @ torch.linalg.inv(ref_ext)       # B x 4 x 4

trans           = calibs.clone()[:, :3, :3]                            # B x 4 x 4
crop_parameters = trans_to_crop_params(trans, intrinsics[:, :3, :3], im_width_height= [512, 512])
rays            = cameras_to_rays(cameras, intrinsics[:, :3, :3], crop_parameters= crop_parameters, return_mode= 'directions_plucker') # B x 3 x H x W

rays  = rays.cpu().float().numpy()

m = 5
r = 1
c = 4
cmap = 'magma_r'
vmin = np.min(rays[:, m])
vmax = np.max(rays[:, m])

plt.figure(figsize= (6*c, 6*r), dpi= params.DPI)
plt.subplot(r,c,1)
plt.imshow(rays[0, m], vmin= vmin, vmax= vmax, cmap= cmap)
plt.axis('off')
plt.title('Height0')
plt.subplot(r,c,2)
plt.imshow(rays[1, m], vmin= vmin, vmax= vmax, cmap= cmap)
plt.axis('off')
plt.title('Height+0.15')
plt.subplot(r,c,3)
plt.imshow(rays[2, m], vmin= vmin, vmax= vmax, cmap= cmap)
plt.axis('off')
plt.title('Height+0.30')
plt.subplot(r,c,4)
plt.imshow(rays[3, m], vmin= vmin, vmax= vmax, cmap= cmap)
plt.axis('off')
plt.title('Height+0.45')

# savefig(plt, "images/plot_plucker_rays.png")
plt.show()
plt.close()