"""
    Sample Run:
    python .py
"""
import os, sys
sys.path.append(os.getcwd())

import numpy as np
np.set_printoptions   (precision= 2, suppress= True)

from plot.common_operations import *
import plot.plotting_params as params
import matplotlib


label_map = {'iou_0_7_easy': r'$AP_{3D}^{0.7}$',
           'iou_0_7_med': r'$AP_{3D}^{0.7}$',
           'iou_0_7_hard': r'$AP_{3D}^{0.7}$',
           'iou_0_5_easy': r'$AP_{3D}^{0.5}$',
           'iou_0_5_med': r'$AP_{3D}^{0.5}$',
           'iou_0_5_hard': r'$AP_{3D}^{0.5}$',
           'mae_easy': 6,
           'mae_med': 7,
           'mae_hard': 8}

height_arr = np.array([0, 6, 12, 18, 24, 30]) * 0.0254
height_arr = np.round(height_arr, 2)
new_height_arr = []
for i, h in enumerate(height_arr):
    new_height_arr.append(str(h) + "\n" + str(i))

lw = params.lw
color1 = params.color_green
color2 = 'gold'
fs         = 28
legend_fs  = 22
ticks_fs   = fs - 4
matplotlib.rcParams.update({'font.size': fs})


oracle = np.array([
              [74.60, 78.34, 74.36, 83.34, 85.65, 86.49],
              [59.79, 60.25, 58.72,	62.25, 64.59, 61.99]
              ])

domain =   np.array([
              [74.60, 74.92, 73.83, 72.42, 66.44, 62.18],
              [59.79, 54.84, 45.61,	31.86, 62.18, 1.38]
              ])

soln_list  = []
color_list = []
style_list = []
label_list = []

func(domain,
     height_arr, new_height_arr, color1, color2, label_map,
     color_list, style_list, ticks_fs, legend_fs, lw,
     oracle= oracle, soln_list= soln_list, label_list= label_list, key = 'iou_0_7_med', multiply= 1, suffix= 'twod', ymax= 88)