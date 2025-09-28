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

from lib.helpers.file_io import read_numpy

def func(npy_path, seman_files):
    a = read_numpy(npy_path)[:, 0].astype(np.float64) # B x 512 x 512
    vmin = 0
    vmax = 1
    cmap = 'magma'

    num_images = a.shape[0]
    b = np.zeros_like(a)
    for i, p in enumerate(seman_files):
        seman = read_numpy(p, show_message= False)
        output = np.zeros_like(seman)
        mask = seman == 1
        output[mask] = 1
        b[i] = output


        if (i) % 10 == 0:
            plt.figure(figsize= (12,6), dpi= params.DPI)
            plt.subplot(121)
            plt.imshow(b[i], cmap= cmap, vmin= vmin, vmax= vmax)
            plt.title('GT')
            plt.axis('off')
            plt.subplot(122)
            plt.imshow(a[i], cmap= cmap, vmin= vmin, vmax= vmax)
            plt.title('Pred Lie Weight')
            plt.axis('off')
            savefig(plt, os.path.join(folder, height, str(i).zfill(6) + ".png"))
            # plt.show()
            plt.close()

        if i > 100:
            break

folder       = "output/gup_carla_lie/result_carla/"
height       = "height0"
npy_path     = os.path.join(folder, height, "val_scale/level_0.npy")
seman_folder = os.path.join("data/carla/", height, "validation/seman")
seman_files  =  sorted(glob.glob(seman_folder + "/*.npy"))

func(npy_path, seman_files)