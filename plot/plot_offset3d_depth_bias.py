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
from lib.datasets.kitti_utils import get_calib_from_file
from lib.helpers.file_io import  read_numpy
from lib.helpers.util import project_3d_points_in_4D_format

base_folder = "output/gup_carla_bkp/result_carla/"
downsample  = 4.0

height_arr = np.array([0, 6, 12, 18, 24, 30]) * 0.0254
height_arr = np.round(height_arr, 2)
new_height_arr = []
for i, h in enumerate(height_arr):
    new_height_arr.append(str(h) + "\n" + str(i))
height_folder_list = ["height" + str(x) for x in np.array([0, 6, 12, 18, 24, 30])]

def metric(pred, gt):
    return np.median(pred-gt)


biases = np.zeros((len(height_folder_list), 3))
oracle = np.zeros((6,))

for i, h in enumerate(height_folder_list):
    P2_old    = get_calib_from_file(os.path.join("data/carla/", h, "validation/calib/000000.txt"))['P2']
    P2        = np.eye(4)
    P2[:3, :] = P2_old
    out_file  = os.path.join(base_folder, h, "pred_gt_box.npy")
    data      = read_numpy(out_file)
    pred_XYZ, gt_XYZ = data[:, :3], data[:, 3:]

    proj_pred_XYZ = project_3d_points_in_4D_format(p2= P2, points_4d= pred_XYZ.transpose(), pad_ones= True).transpose()[:, :3]
    proj_gt_XYZ   = project_3d_points_in_4D_format(p2= P2, points_4d=   gt_XYZ.transpose(), pad_ones= True).transpose()[:, :3]

    biasX = metric(proj_pred_XYZ[:,0], proj_gt_XYZ[:,0])/downsample
    biasY = metric(proj_pred_XYZ[:,1], proj_gt_XYZ[:,1])/downsample
    biasZ = metric(proj_pred_XYZ[:,2], proj_gt_XYZ[:,2])

    print("MAE X = {:.2f} {:.2f} {:.2f}".format(biasX, biasY, biasZ))

    biases[i] = np.array([biasX, biasY, biasZ])

plt.figure(figsize= params.size, dpi= params.DPI)
plt.plot(height_arr, biases[:, 0], label='u Median (px)', lw= params.lw, c= params.color_seaborn_1)
plt.plot(height_arr, biases[:, 1], label='v Median (px)', lw= params.lw, c= params.color_seaborn_3)
plt.plot(height_arr, biases[:, 2], label='Z Median (m)', lw= params.lw, c= 'gold')
plt.plot(height_arr, oracle, label='Oracle', lw= params.lw, c= params.color_green)
plt.xlabel(r'$\Delta$' + 'Height (m)')
plt.ylabel('Error' + r' $(Pred-GT)$')
plt.xlim(left= -0.02)
plt.ylim(-2.2, 2.2)
plt.grid(True)
plt.xticks(height_arr, labels=new_height_arr)
plt.legend(loc='lower left', borderaxespad= params.legend_border_axes_plot_border_pad, borderpad= params.legend_border_pad, labelspacing= params.legend_vertical_label_spacing, handletextpad= params.legend_marker_text_spacing)
savefig(plt, "images/offset3d_depth_bias.png")
plt.close()