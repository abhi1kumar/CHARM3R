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


domain =   np.array([
              [42.39, 27.01, 9.85, 1.03, 0.02, 0],
              [59.79, 54.84, 45.61,	31.86, 12.15, 1.38]
              ])

oracle = np.array([
              [42.39, 43.06, 41.23, 44.18, 45.61, 43.65],
              [59.79, 60.25, 58.72,	62.25, 64.59, 61.99]
              ])

gup_carla_0_24 = np.array([[
                3.21, 3.21, 3.32, 3.87, 4.01, 3.98
                ], [
                26.49, 26.49, 28.81, 30.18, 30.81, 31.23
]])


gup_carla_0_18 = np.array([[
                3.35, 3.31, 3.48, 4.1, 3.25, 2.28
                ], [
                25.18, 26.9, 28.75, 31.28, 30.26, 30.43
]])

gup_carla_0_18_75k = np.array([[
                49.13, 49.72, 51.96, 50.89, 48.32, 48.37
                ], [
                63.7, 64.04, 64.81, 66.52, 67.04, 68.1
]])

gup_carla_0_18_50k = np.array([[
                42.99, 46.39, 46.57, 47.55, 46.87, 44.31
                ], [
                56.72, 60.14, 61.49, 65.3, 66.91, 65.53
]])

gup_carla_0_18_25k = np.array([[
                38.20, 37.64, 37.75, 39.23, 38.49, 33.46
                ],[
                56.21, 57.14, 57.60,  59.00, 59.50, 53.51
]])

gup_carla_0_12  = np.array([
              [47.53, 47.56, 47.44, 48.16, 43.44, 38.85],
              [59.24, 59.39, 61.39,	63.44, 64.48, 65.55]
])

gup_carla_0_6 = np.array([
              [40.83, 42.82, 38.99, 37.44, 28.81, 17.15],
              [55.50, 55.60, 56.84,	56.19, 53.35, 49.99]
])

gup_carla_0_12_25k = np.array([
              [34.71, 37.98, 38.49,	36.71, 31.53, 27.63],
              [53.19, 58.49, 55.49,	55.96, 54.13, 51.64]
])

gup_carla_0_6_25k  = np.array([
              [40.46, 43.13, 38.95,	35.18, 27.55, 18.65],
              [58.41, 61.35, 59.79,	58.41, 54.65, 51.92]
])

# ==================================================================================================
# All data
# ==================================================================================================

soln_list  = [gup_carla_0_24, gup_carla_0_18, gup_carla_0_12, gup_carla_0_6]
color_list = [params.color_seaborn_1, params.color_seaborn_0, params.color_seaborn_15, params.color_seaborn_5]
style_list = ['solid', 'solid', 'solid', 'solid']
label_list = ['Ht 0+1+2+3+4 5x', 'Ht 0+1+2+3 4x', 'Ht 0+1+2 3x', 'Ht 0+1 2x']

func(domain,
     height_arr, new_height_arr, color1, color2, label_map,
     color_list, style_list, ticks_fs, legend_fs, lw,
     oracle= oracle, soln_list= soln_list, label_list= label_list, key = 'iou_0_7_med', multiply= 1, suffix= 'performance_aug_all_data')
func(domain,
     height_arr, new_height_arr, color1, color2, label_map,
     color_list, style_list, ticks_fs, legend_fs, lw,
     oracle= oracle, soln_list= soln_list, label_list= label_list, key = 'iou_0_5_med', multiply= 1, suffix= 'performance_aug_all_data')

# ==================================================================================================
# Inc data
# ==================================================================================================
soln_list  = [gup_carla_0_18, gup_carla_0_18_75k, gup_carla_0_18_50k, gup_carla_0_18_25k]
color_list = [params.color_seaborn_0, params.color_seaborn_1, params.color_seaborn_15, params.color_seaborn_5]
style_list = ['solid', 'solid', 'solid', 'solid']
label_list = ['Ht 0+1+2+3+4 4x', 'Ht 0+1+2+3+4 3x', 'Ht 0+1+2+3+4 2x', 'Ht 0+1+2+3+4 1x']

func(domain,
     height_arr, new_height_arr, color1, color2, label_map,
     color_list, style_list, ticks_fs, legend_fs, lw,
     oracle= oracle, soln_list= soln_list, label_list= label_list, key = 'iou_0_7_med', multiply= 1, suffix= 'performance_aug_inc_data')
func(domain,
     height_arr, new_height_arr, color1, color2, label_map,
     color_list, style_list, ticks_fs, legend_fs, lw,
     oracle= oracle, soln_list= soln_list, label_list= label_list, key = 'iou_0_5_med', multiply= 1, suffix= 'performance_aug_inc_data')

# ==================================================================================================
# Same data
# ==================================================================================================
soln_list  = [gup_carla_0_18_25k, gup_carla_0_12_25k, gup_carla_0_6_25k]
color_list = [params.color_seaborn_0, params.color_seaborn_15, params.color_seaborn_5]
style_list = ['solid', 'solid', 'solid', 'solid']
label_list = ['Ht 0+1+2+3 1x', 'Ht 0+1+2 1x', 'Ht 0+1 1x']

func(domain,
     height_arr, new_height_arr, color1, color2, label_map,
     color_list, style_list, ticks_fs, legend_fs, lw,
     oracle= oracle, soln_list= soln_list, label_list= label_list, key = 'iou_0_7_med', multiply= 1, suffix= 'performance_aug_same_data')
func(domain,
     height_arr, new_height_arr, color1, color2, label_map,
     color_list, style_list, ticks_fs, legend_fs, lw,
     oracle= oracle, soln_list= soln_list, label_list= label_list, key = 'iou_0_5_med', multiply= 1, suffix= 'performance_aug_same_data')