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

folder = "data/coda/"
split = ["training_org", "testing_org"]

from lib.helpers.file_io import read_lines, write_lines
from lib.helpers.math_3d import convertRot2Alpha
for s in split:
    label_folder = os.path.join(folder, s, "label_2")
    label_files  = sorted(glob.glob(label_folder + "/*.txt"))
    for i, l in enumerate(label_files):
        output_boxes_text = []
        boxes_text = read_lines(l, strip= False)
        for box in boxes_text:
            params = box.split(" ")
            ry3d = float(params[-1].strip('\n'))
            z3d  = float(params[-2])
            x3d  = float(params[-4])
            params[3] = str(np.round(convertRot2Alpha(ry3d, z3d, x3d), 2))
            output_boxes_text.append(" ".join(params))
        write_lines(l, output_boxes_text)
        if (i+1) % 1000 == 0 or (i+1) == len(label_files):
            print("{} labels done...".format(i+1))
print("Done...")