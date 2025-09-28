"""
    Sample Run:
    python .py
"""
import os, sys

import matplotlib.pyplot as plt

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

from lib.helpers.ground_plane import ground_depth
from lib.datasets.kitti_utils import get_calib_from_file
from lib.helpers.file_io import imread, read_numpy

device = torch.device("cuda")
B = 3
key = "000040"
downsample    = 1
r = 1
c = 4
vmin = 0
vmax = 50
cmap = 'magma_r'
height = "height0"

calib_path = "data/carla/" + height + "/training/calib/" + key + ".txt"
image_path = "data/carla/" + height + "/training/image/" + key + ".jpg"
depth_path = "data/carla/" + height + "/training/depth/" + key + ".npy"
image = imread(image_path)
depth = read_numpy(depth_path)

info          = get_calib_from_file(calib_path)
intrinsics    = torch.tensor(info["intrinsics"]).to(device).unsqueeze(0).repeat(B, 1, 1)
gd_to_cam     = torch.tensor(info["gd_to_cam"]).to(device).unsqueeze(0).repeat(B, 1, 1)    # B x 3 x 4
extrinsics    = torch.eye(4).to(device).unsqueeze(0).repeat(B, 1, 1)                       # B x 4 x 4

trans         = torch.eye(3).to(device).unsqueeze(0).repeat(B, 1, 1)        # B x 3 x 3
cam_height    = gd_to_cam[:, 1, 3]     # B
gedepth       = ground_depth(intrinsics, extrinsics, cam_height, im_h= image.shape[0], im_w= image.shape[1], downsample= downsample, trans= trans)
ge            = gedepth[0].cpu().float().numpy()
ge[ge < 0]    = 1000
ge[image.shape[0]//2,:] = 1000

fig= plt.figure(figsize= (6*c,6), dpi= params.DPI)
plt.subplot(r,c,1)
plt.imshow(image)
plt.axis('off')
plt.title('Image')

plt.subplot(r,c,2)
im = plt.imshow(depth, vmin= vmin, vmax= vmax, cmap=cmap)
plt.axis('off')
plt.title('Depth (m)')
ax = plt.gca()
cax = fig.add_axes([ax.get_position().x1+0.001,ax.get_position().y0,0.01,ax.get_position().height])
plt.colorbar(im, cax=cax)

plt.subplot(r,c,3)
plt.imshow(ge, vmin= vmin, vmax= vmax, cmap=cmap)
plt.axis('off')
plt.title('Ground (m)')

plt.subplot(r,c,4)
im = plt.imshow((ge-depth), vmin= -2, vmax= 2, cmap= 'PiYG')
plt.axis('off')
plt.title('GD - Depth (m)')
ax = plt.gca()
cax = fig.add_axes([ax.get_position().x1+0.001,ax.get_position().y0,0.01,ax.get_position().height])
plt.colorbar(im, cax=cax)

savefig(plt, "images/test_ground_depth_compare.png")
plt.close()

fig= plt.figure(figsize= (6,6), dpi= params.DPI)
plt.imshow(ge, vmin= vmin, vmax= vmax, cmap=cmap)
plt.axis('off')
savefig(plt, "images/test_ground_depth.png")
plt.show()
plt.close()