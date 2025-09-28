"""
    Sample Run:
    python test/plot_bottom_center_variation.py

    Run the following to get the stats.
    python test/get_oracle_with_ground_depth.py
"""
import os, sys
sys.path.append(os.getcwd())

import os.path as osp
import glob
import numpy as np
import torch
import torch.nn as nn
import random

np.set_printoptions   (precision= 2, suppress= True)
torch.set_printoptions(precision= 2, sci_mode= False)

from plot.common_operations import *
import plot.plotting_params as params
import matplotlib

from lib.helpers.file_io import read_numpy
from scipy import stats
from sklearn.linear_model import LinearRegression
from skimage.metrics import mean_squared_error

fs         = 20
matplotlib.rcParams.update({'font.size': fs})

def do_stats(x, y):
    # corr = stats.pearsonr(x, y)[0]
    # slope, intercept, r, p, std_err = stats.linregress(x, y)
    # return corr, slope, intercept, r, p, std_err

    flag = False
    if x.ndim == 1:
        flag = True
        corr = stats.pearsonr(x, y)[0]
        x = x[:, np.newaxis]
    else:
        corr = stats.pearsonr(x[:, 0], y)[0]

    model = LinearRegression()
    model.fit(x, y)
    slope = model.coef_
    intercept = model.intercept_
    # Evaluate the model
    y_pred = model.predict(x)
    mse    = mean_squared_error(y, y_pred)
    rmse   = np.sqrt(mse)

    if flag:
        slope = slope[0]

    return corr, slope, intercept, 0.0, 0.0, rmse


def plotter(npy_path):
    output_path = "images/bottom_center_" + os.path.basename(npy_path).replace("_all_data", "").replace(".npy", ".png")

    if "kitti" in npy_path.lower():
        sind = 0
        TH   = 370
    elif "coda" in npy_path.lower():
        sind = 0
        TH   = 500
    elif "waymo" in npy_path.lower():
        # shift index by 1
        sind = 1
        TH   = 832
    else:
        sind = 0
        TH   = 512

    boxes    = read_numpy(npy_path)
    boxes    = boxes.astype(np.float32)
    inlier   = np.logical_and(boxes[:, 19+sind] >= -10, boxes[:, 19+sind] < TH)

    boxes    = boxes[inlier]

    #        0   1    2     3   4   5  6    7    8    9    10   11   12   13    14    15  16  17  18     19           20
    # (cls, -1, -1, alpha, x1, y1, x2, y2, h3d, w3d, l3d, x3d, y3d, z3d, ry3d, score, cx, cy, bx, by, by_ideal, depth_center)

    h2d   = boxes[:, 6] - boxes[:, 4]
    u     = 0.5 * (boxes[:, 5] + boxes[:, 3])
    v     = 0.5 * (boxes[:, 6] + boxes[:, 4])

    # 3D properties
    depth_old = boxes[:, 12]
    h3d = boxes[:, 7]
    XYZ = boxes[:, 10:13]  # N x 3
    XYZ[:, 1] -= h3d / 2.0

    cx       = boxes[:, 15+sind]
    cy       = boxes[:, 16+sind]
    center_diff = cy - v
    bx       = boxes[:, 17+sind]
    by       = boxes[:, 18+sind]
    by_ideal = boxes[:, 19+sind]

    r = 1
    c = 2

    add_to_by  = by_ideal - by

    x_all = np.concatenate((h2d[:, np.newaxis], center_diff[:, np.newaxis]), axis= 1)
    corr, slope, intercept, _, _, err = do_stats(x_all, add_to_by)
    print('E= {:.2f} Sl= {:.1f} {:.1f} Int= {:.1f} Cor= {:.1f}'.format(err, slope[0], slope[1], intercept, corr))
    corr, slope, intercept, _ , _, err = do_stats(h2d, add_to_by)
    out1 = 'E= {:.2f} Sl= {:.1f} Int= {:.1f} Cor= {:.1f}'.format(err, slope, intercept, corr)
    print(out1)
    corr, slope, intercept, _ , _, err = do_stats(center_diff, add_to_by)
    out2 = 'E= {:.2f} Sl= {:.1f} Int= {:.1f} Cor= {:.1f}'.format(err, slope, intercept, corr)
    print(out2)

    plt.figure(figsize= (8*c, 6*r), dpi= params.DPI)

    plt.subplot(r,c,1)
    plt.scatter(h2d, add_to_by, label= 'Calc', c= [params.color_seaborn_0])
    plt.xlabel('$h_{2D}$')
    plt.ylabel(r'$\Delta$Pixel')
    plt.xlim(0, 400)
    plt.ylim(-200, 200)
    plt.title(out1)
    plt.grid()

    plt.subplot(r,c,2)
    plt.scatter(center_diff, add_to_by, label= 'Calc', c= [params.color_seaborn_1])
    plt.xlabel(r'$\Delta$Cen$_y$')
    plt.ylabel(r'$\Delta$Pixel')
    plt.xlim(-60, 20)
    plt.ylim(-200, 200)
    plt.title(out2)
    plt.grid()

    savefig(plt, output_path)
    # plt.show()
    plt.close()



# ==================================================================================================
# Main starts here
# ==================================================================================================
npy_path    = "output/oracle_ground/CARLA_height0_all_data.npy"
plotter(npy_path)
npy_path    = "output/oracle_ground/CARLA_height30_all_data.npy"
plotter(npy_path)
npy_path    = "output/oracle_ground/KITTI_all_data.npy"
plotter(npy_path)
npy_path    = "output/oracle_ground/Coda_all_data.npy"
plotter(npy_path)
npy_path    = "output/oracle_ground/Waymo_all_data.npy"
plotter(npy_path)
