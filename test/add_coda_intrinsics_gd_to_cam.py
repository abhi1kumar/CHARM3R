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

from lib.datasets.kitti_utils import get_calib_from_file
from lib.helpers.file_io import read_lines, write_lines

def format_one_matrix(t):
    def float_formatter(x):
        return f"{x:.12e}"
    t = t[:3, :].flatten()
    return np.array2string(t, formatter={'float_kind': float_formatter}).replace("\n", "").replace("[", "").replace("]", "")

folder = "data/coda/"
split = ["training_org", "testing_org"]

for s in split:
    calib_folder = os.path.join(folder, s, "calib")
    calib_files  = sorted(glob.glob(calib_folder + "/*.txt"))
    for c in calib_files:
        calib_text = read_lines(c, strip= False)

        P2 = get_calib_from_file(c)['P2']
        intrinsics      = P2[:3, :3]
        gd_to_cam       = np.eye(4)
        gd_to_cam[1, 3] = 0.75      # See Fig. 2 of https://arxiv.org/pdf/2309.13549

        calib_text.append("gd_to_cam: " + format_one_matrix(gd_to_cam) + "\n")
        calib_text.append("intrinsics: " + format_one_matrix(intrinsics) + "\n")

        output_calib_path = c #osp.join("/home/abhinav/Desktop/hello", osp.basename(c))
        write_lines(path= output_calib_path, lines_with_return_character= calib_text)