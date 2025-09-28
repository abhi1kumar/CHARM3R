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


from lib.helpers.unidrive import run_project_pixels_cuda
from lib.datasets.kitti_utils import get_calib_from_file
from lib.helpers.file_io import read_image

ONES = np.array([[0.,0.,0.,1.]])

height = "height0"
new_height = "height30"
key    = "000040"
split  = "/training/"
folder = "data/carla/" + height + split
new_folder = "data/carla/" + new_height + split

ifile  = osp.join(folder, "image/" + key + ".jpg")
cfile  = osp.join(folder, "calib/" + key + ".txt")
image  = read_image(ifile)
calib  = get_calib_from_file(cfile)
original_extrinsics = calib['gd_to_cam']
original_extrinsics[1, 3] -= original_extrinsics[1, 3]
original_extrinsics = np.concatenate((original_extrinsics, ONES), axis= 0)


ifile  = osp.join(new_folder, "image/" + key + ".jpg")
cfile  = osp.join(new_folder, "calib/" + key + ".txt")
new_image  = read_image(ifile)
new_calib  = get_calib_from_file(cfile)

ONES = np.array([[0.,0.,0.,1.]])
intrinsics = calib['intrinsics']
new_extrinsics = new_calib['gd_to_cam']
new_extrinsics[1, 3] -= 1.51
new_extrinsics[1, 3] *= -1
new_extrinsics = np.concatenate((new_extrinsics, ONES), axis= 0)
output_image  = run_project_pixels_cuda(new_image, intrinsics, new_extrinsics, original_extrinsics, Dis= 50)
output_image2 = run_project_pixels_cuda(new_image, intrinsics, new_extrinsics, original_extrinsics, Dis= 3.14)

r= 1
c= 4
plt.figure(figsize= (6*c, 6), dpi= params.DPI)
plt.subplot(r,c,1)
plt.imshow(new_image)
plt.axis('off')
plt.title('Ht5')

plt.subplot(r,c,2)
plt.imshow(output_image)
plt.axis('off')
plt.title('UniDrive')

plt.subplot(r,c,3)
plt.imshow(output_image2)
plt.axis('off')
plt.title('UniDrive++')

plt.subplot(r,c,4)
plt.imshow(image)
plt.axis('off')
plt.title('Ht0')

savefig(plt, "images/test_unidrive_project.png")
plt.close()

c= 3
plt.figure(figsize= (6*c, 6), dpi= params.DPI)
plt.subplot(r,c,1)
plt.imshow(new_image)
plt.axis('off')
plt.title('Ht5')

plt.subplot(r,c,2)
plt.imshow(output_image)
plt.axis('off')
plt.title('UniDrive')

plt.subplot(r,c,3)
plt.imshow(image)
plt.axis('off')
plt.title('Ht0')

savefig(plt, "images/test_unidrive_project_0.png")
plt.close()