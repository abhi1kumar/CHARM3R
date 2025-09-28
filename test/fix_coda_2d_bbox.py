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
from lib.helpers.file_io import read_lines, write_lines
from lib.datasets.kitti_utils import get_calib_from_file
from lib.helpers.math_3d import project_3d
from lib.helpers.math_3d import convertRot2Alpha

W = 1224
H = 1024

folder = "data/coda/"
split = ["training_org", "testing_org"]

for s in split:
    label_folder = os.path.join(folder, s, "label_2")
    calib_folder = os.path.join(folder, s, "calib")
    label_files  = sorted(glob.glob(label_folder + "/*.txt"))
    calib_files  = sorted(glob.glob(calib_folder + "/*.txt"))
    print("Label_folder= {}".format(label_folder))

    for i, (l,c) in enumerate(zip(label_files, calib_files)):
        p2_3     = get_calib_from_file(c)['P2']  # 3 x 4
        p2       = np.eye(4)
        p2[:3]   = p2_3

        gt_img   = read_csv(l, ignore_warnings= True, use_pandas= True)
        if gt_img is None or gt_img.shape[0] <= 0:
            continue
        gt_class =  gt_img[:, 0]
        gt_box   =  gt_img[:, 1:]
        gt_box   =  np.array(gt_box).astype(np.float32)

        # 3d bbox
        #       0  1   2      3   4   5   6    7    8    9   10   11   12    13     14
        # cls, -1, -1, alpha, x1, y1, x2, y2, h3d, w3d, l3d, x3d, y3d, z3d, ry3d, lidar
        h3d = gt_box[:, 7]
        w3d = gt_box[:, 8]
        l3d = gt_box[:, 9]
        x3d = gt_box[:, 10]
        y3d = gt_box[:, 11] - h3d/2
        z3d = gt_box[:, 12]
        r3d = gt_box[:, 13]
        vert2d = project_3d(p2, x3d, y3d, z3d, w3d, h3d, l3d, r3d, return_3d=False) # N x 4 x 8
        x1  = np.min(vert2d[:, 0], axis= 1) # N
        x2  = np.max(vert2d[:, 0], axis= 1)
        y1  = np.min(vert2d[:, 1], axis= 1)
        y2  = np.max(vert2d[:, 1], axis= 1)

        # Projected 2d Box clipped
        x1  = np.maximum(x1, np.zeros_like(x1))
        x2  = np.minimum(x2, (W-1) * np.ones_like(x1))
        y1  = np.maximum(y1, np.zeros_like(x1))
        y2  = np.minimum(y2, (H-1) * np.ones_like(x1))

        output_boxes_text = []
        boxes_text = read_lines(l, strip= False)
        for b, box in enumerate(boxes_text):
            params = box.split(" ")
            #   0   1    2     3   4   5  6    7    8    9    10   11   12         13    14   15
            # (cls, -1, -1, alpha, x1, y1, x2, y2, h3d, w3d, l3d, x3d, y3d + h/2, z3d, ry3d, score,


            # ry3d_b = float(params[-1].strip('\n'))
            # z3d_b  = float(params[-2])
            # x3d_b  = float(params[-4])
            #params[3] = str(np.round(convertRot2Alpha(ry3d_b, z3d_b, x3d_b), 2))

            occ = float(params[1])
            # Some boxes are too big in default coda converter
            area = (x2[b]-x1[b])*(y2[b]-y1[b])
            if area >= 0.7*H*W:
                occ = 1.0

            if occ == 0.0:
                params[1] = '0.00'
            elif occ == 1.0:
                params[1] = '1.00'
            else:
                params[1] = str(np.round(occ, 2))

            if occ == 1.0:
                params[4] = '0.0'
                params[5] = '0.0'
                params[6] = '0.0'
                params[7] = '0.0'
            elif z3d[b] > 0.1:
                params[4] = str(np.round(x1[b], 2))
                params[5] = str(np.round(y1[b], 2))
                params[6] = str(np.round(x2[b], 2))
                params[7] = str(np.round(y2[b], 2))

            output_boxes_text.append(" ".join(params))

        # output_path = osp.join("/home/abhinav/Desktop/coda/training_org/label_2", osp.basename(l))
        output_path = l
        write_lines(output_path, output_boxes_text)
        if (i+1) % 1000 == 0 or (i+1) == len(label_files):
            print("{} labels done...".format(i+1))
    print("Done...")
