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
print(height_arr)
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
              [59.79, 54.84, 45.61, 31.86, 12.15, 1.38]
              ])

oracle = np.array([
              [42.39, 43.06, 41.23, 44.18, 45.61, 43.65],
              [59.79, 60.25, 58.72, 62.25, 64.59, 61.99]
              ])

oracle_bev = np.array([[
0, 0, 0, 0, 0, 0
                                  ], [
62.18, 60.75, 61.36, 64.64, 67.03, 64.59
]])

gup_carla_height6  = np.array([[
              24.76, 43.06, 30.78, 12.45, 1.18, 0
              ],[
              50.91, 60.25, 54.36, 49.76, 37.29, 19.38
]])

gup_carla_height12  = np.array([[
              10.61, 23.05, 41.23, 30.1, 14.33, 3.56
              ],[
              38.61, 51.37, 58.72, 59.31, 51.02, 43.32
]])

tr_12_time1x  = np.array([
              [31.45, 39.64, 41.9, 39.7, 35.01, 28.9],
              [54.24, 58.34, 62.3, 61.3, 59.81, 59.7]
])

lie = np.array([[
            39.97,21.44, 6.08, 0.49, 0.03, 0.01
            ],[
            55.08,48.35,36.41,24.03,8.64,1.17
]])

gup_coord = np.array([[
            37.35, 25.41, 11.01, 1.41, 0.06, 0.01
            ],[
            53.96, 47.66, 41.76, 32.04, 20.51, 8.61
]])

gup_pp = np.array([[
            42.39, 34.15, 24.05, 13.47, 4.72, 0.98
            ], [
            59.79, 56.62, 52.31, 47.62, 36.01, 21.72
]])

gup_bev = np.array([[
0, 0, 0, 0, 0, 0
                                  ], [
62.18, 55.92, 48.77, 38.38, 22.36, 9.29
]])

gup_pp_bev = np.array([[
0, 0, 0, 0, 0, 0
                                  ], [
62.18, 59.21, 55.27, 51.42, 43.66, 30.11
]])

gup_coord_pp = np.array([[
            37.35, 30.86, 21.23, 10.31, 3.26, 0.38
            ], [
            53.96, 49.26, 44.22, 40.64, 32.43, 18.41
]])

lie_pp = np.array([[
            39.97, 33.12,21.01,12.68,4.68,0.86
            ], [
            55.08,54.94,48.1,46.89,36.52,25.84
]])

run_004 = np.array([[
    38.82,24.74,7.65,0.34,0.02,0.0,
    ], [
    57.98,52.36,44.04,30.33,9.24,0.61,
]])

run_004_pp = np.array([[
    38.82,32.88,20.05,8.83,1.67,0.12,
    ], [
    57.98,53.88,49.09,39.65,25.48,9.51
]])

gup_coord_inverted = np.array([[
            42.39, 29.38, 11.59, 1.87, 0.16, 0.01
            ], [
            59.79, 51.85, 49.13, 38.4, 21.16, 6
]])

gup_coord_inverted_pp = np.array([[
            37.35, 32.04, 23.67, 14.96, 5.25, 1.21
            ], [
            53.96, 50.69, 51, 50.05, 41.77, 27.17
]])

lie_coord_inverted = np.array([[
            42.39, 26.42, 7.56, 0.72, 0.04, 0.01
            ], [
            59.79, 49.41, 40.92, 27.51, 10.65, 2.33
]])

lie_coord_inverted_pp = np.array([[
            37.35, 30.26, 19.51, 9.83, 2.72, 0.42
            ], [
            53.96, 50.5, 45.4, 37.51, 23.74, 11.28
]])

gup_ppv2 = np.array([[
            42.39, 32.05, 19.08, 10.08, 4.29, 1.24
            ], [
            59.79, 56.66, 50.87, 43.54, 33.61, 23.41
]])

gup_ppv1_uvz = np.array([[
            42.39, 34.69, 25.98, 18.09, 9.22, 2.42
            ], [
            59.79, 56.6, 52.01, 47.43, 35.69, 22.75
]])

gup_npe_pp = np.array([[
            42.39, 31.26, 21.12, 10.9, 2.16, 0.2,
            ], [
            59.79, 54.11, 49.71, 43.15, 32.04, 19.37
]])

gup_fourier_pp = np.array([[
            42.39, 29.47, 20.09, 8.17, 1.7, 0.24
            ], [
            59.79, 51.12, 46.54, 42.78, 32.41, 19.59
]])

gup_oracle_depth = np.array([[
            0, 0, 0, 0, 0, 0
            ], [
            35.09, 31.06, 28.05, 25.25, 22.74, 15.12
]])

gup_oracle_inv_depth = np.array([[
            0, 0, 0, 0, 0, 0
            ], [
            61.51, 56.49, 51.97, 48.23, 38.48, 24.59
]])

gup_oracle_normal = np.array([[
            0, 0, 0, 0, 0, 0
            ], [
            80.03, 80.14, 75.46, 71.02, 62.2, 39.62
]])

gup_oracle_normal_inv_depth = np.array([[0, 0, 0, 0, 0, 0
            ], [
            79.69, 73.95, 68.66, 61.84, 49.91, 24.56
]])

gup_oracle_normal_fl = np.array([[0, 0, 0, 0, 0, 0
            ], [
            80.13, 79.36, 75.82, 73.73, 66.14, 51.79
]])
gup_oracle_normal_all = np.array([[0, 0, 0, 0, 0, 0
            ], [
80.28, 79.56, 76.89, 72.29, 64.34, 51.19
]])

gup_kprpe = np.array([[0, 0, 0, 0, 0, 0
            ], [
54.79,50.31,45.93,42.31,29.39,12.88
]])

gup_dsine_normal_fl = np.array([[0, 0, 0, 0, 0, 0
                                 ], [
73.01,70.24,63.12,58.53,47.15, 31.41
]])

gup_metnor_normal_fl = np.array([[0, 0, 0, 0, 0, 0
                                  ], [
73.23,69.29,61.26,57.76,46.15, 29.7
]])

gup_dsine_normal_bev = np.array([[
0, 0, 0, 0, 0, 0
                                  ], [
73.72, 73.26, 69.02, 65.21, 58.31, 50.87
]])

gup_dsine_normal_transf = np.array([[
0, 0, 0, 0, 0, 0
                                  ], [
73.01, 66.14, 52.93, 36.83, 21, 11.46
]])

gup_dsine_normal_transf_bev = np.array([[
0, 0, 0, 0, 0, 0
                                  ], [
73.72, 71.62, 69.47, 66.9, 63.3, 57.5
]])

gup_directions_plucker = np.array([[
0, 0, 0, 0, 0, 0
                                  ], [
73.7, 70.48, 66.4, 58.31, 49.28, 34.93
]])

gup_plucker = np.array([[
0, 0, 0, 0, 0, 0
                                  ], [
74.3, 68.78, 63.21, 56.48, 43.11, 18.38
]])

gup_directions = np.array([[
0, 0, 0, 0, 0, 0
                                  ], [
74.37, 67.9, 63.67, 55.27, 42.8, 24.93
]])

gup_normal = np.array([[
0, 0, 0, 0, 0, 0
                                  ], [
82.72, 79.78, 77.39, 73.08, 67.47, 56.39
]])

edge_

"""
# ==================================================================================================
# Teaser
# ==================================================================================================
soln_list  = []
color_list = []
style_list = []
label_list = []

func(domain,
     height_arr, new_height_arr, color1, color2, label_map,
     color_list, style_list, ticks_fs, legend_fs, lw,
     oracle= oracle, soln_list= soln_list, label_list= label_list, key = 'iou_0_7_med', multiply= 1, suffix= 'performance_teaser')
func(domain,
     height_arr, new_height_arr, color1, color2, label_map,
     color_list, style_list, ticks_fs, legend_fs, lw,
     oracle= oracle, soln_list= soln_list, label_list= label_list, key = 'iou_0_5_med', multiply= 1, suffix= 'performance_teaser')

# ==================================================================================================
# One height
# ==================================================================================================
soln_list  = [gup_carla_height6, gup_carla_height12]
color_list = [params.color_seaborn_0, params.color_seaborn_15]
style_list = ['solid', 'solid']
label_list = ['Ht 1 1x', 'Ht 2 1x']

func(domain,
     height_arr, new_height_arr, color1, color2, label_map,
     color_list, style_list, ticks_fs, legend_fs, lw,
     oracle= oracle, soln_list= soln_list, label_list= label_list, key = 'iou_0_7_med', multiply= 1, suffix= 'performance_one_height')
func(domain,
     height_arr, new_height_arr, color1, color2, label_map,
     color_list, style_list, ticks_fs, legend_fs, lw,
     oracle= oracle, soln_list= soln_list, label_list= label_list, key = 'iou_0_5_med', multiply= 1, suffix= 'performance_one_height')


# ==================================================================================================
# Solutions
# ==================================================================================================
soln_list  = [lie_pp, lie, gup_coord_pp, gup_coord, gup_pp]
color_list = [params.color_seaborn_0, params.color_seaborn_0, params.color_seaborn_15, params.color_seaborn_15, 'gold']
label_list = ['Lie PPv1', 'Lie', 'Coord PPv1', 'Coord', 'Ht 0 PPv1']
style_list = ['solid'] * len(soln_list)
style_list = ['dotted', 'solid', 'dotted', 'solid', 'dotted']

func(domain,
     height_arr, new_height_arr, color1, color2, label_map,
     color_list, style_list, ticks_fs, legend_fs, lw,
     oracle= oracle, soln_list= soln_list, label_list= label_list, key = 'iou_0_7_med', multiply= 1, suffix= 'performance')
func(domain,
     height_arr, new_height_arr, color1, color2, label_map,
     color_list, style_list, ticks_fs, legend_fs, lw,
     oracle= oracle, soln_list= soln_list, label_list= label_list, key = 'iou_0_5_med', multiply= 1, suffix= 'performance')

# ==================================================================================================
# With layer
# ==================================================================================================
soln_list  = [lie_pp, lie, run_004_pp, run_004, gup_pp]
color_list = [params.color_seaborn_0, params.color_seaborn_0, params.color_seaborn_15, params.color_seaborn_15, 'gold']
label_list = ['First PPv1', 'First', 'Last PPv1', 'Last', 'Ht 0 PPv1']
style_list = ['dotted', 'solid', 'dotted', 'solid', 'dotted']

func(domain,
     height_arr, new_height_arr, color1, color2, label_map,
     color_list, style_list, ticks_fs, legend_fs, lw,
     oracle= oracle, soln_list= soln_list, label_list= label_list, key = 'iou_0_7_med', multiply= 1, suffix= 'performance_layer')
func(domain,
     height_arr, new_height_arr, color1, color2, label_map,
     color_list, style_list, ticks_fs, legend_fs, lw,
     oracle= oracle, soln_list= soln_list, label_list= label_list, key = 'iou_0_5_med', multiply= 1, suffix= 'performance_layer')

# ==================================================================================================
# With Post processing
# ==================================================================================================
soln_list  = [gup_ppv2, gup_pp]
color_list = ['gold', 'gold']
label_list = ['Ht 0 PPv2', 'Ht 0 PPv1']
style_list = ['dashed', 'dotted']

func(domain,
     height_arr, new_height_arr, color1, color2, label_map,
     color_list, style_list, ticks_fs, legend_fs, lw,
     oracle= oracle, soln_list= soln_list, label_list= label_list, key = 'iou_0_7_med', multiply= 1, suffix= 'performance_with_pp')
func(domain,
     height_arr, new_height_arr, color1, color2, label_map,
     color_list, style_list, ticks_fs, legend_fs, lw,
     oracle= oracle, soln_list= soln_list, label_list= label_list, key = 'iou_0_5_med', multiply= 1, suffix= 'performance_with_pp')

# ==================================================================================================
# With (u,v,Z) Post processing
# ==================================================================================================
soln_list  = [gup_ppv1_uvz, gup_pp]
color_list = [params.color_seaborn_0, 'gold']
label_list = ['Ht 0 PP uvZ', 'Ht 0 PPv1']
style_list = ['solid', 'dotted']

func(domain,
     height_arr, new_height_arr, color1, color2, label_map,
     color_list, style_list, ticks_fs, legend_fs, lw,
     oracle= oracle, soln_list= soln_list, label_list= label_list, key = 'iou_0_7_med', multiply= 1, suffix= 'performance_with_UVZ')
func(domain,
     height_arr, new_height_arr, color1, color2, label_map,
     color_list, style_list, ticks_fs, legend_fs, lw,
     oracle= oracle, soln_list= soln_list, label_list= label_list, key = 'iou_0_5_med', multiply= 1, suffix= 'performance_with_UVZ')


# ==================================================================================================
# With Camera encodings
# ==================================================================================================
soln_list  = [lie_coord_inverted_pp, lie_coord_inverted, lie_pp, lie, gup_coord_inverted_pp, gup_coord_inverted, run_004_pp, run_004, gup_pp]
color_list = [params.color_seaborn_0, params.color_seaborn_1, params.color_seaborn_0, params.color_seaborn_0, params.color_seaborn_15, params.color_seaborn_3, params.color_seaborn_15, params.color_seaborn_15, 'gold']
label_list = ['Lie Inv PPv1', 'Lie Inv', 'Lie PPv1', 'Lie', 'Coord Inv PPv1', 'Coord Inv', 'Coord PPv1', 'Coord', 'Ht 0 PPv1']
style_list = ['dashed', 'solid', 'dotted', 'solid', 'dashed', 'solid', 'dotted', 'solid', 'dotted']

func(domain,
     height_arr, new_height_arr, color1, color2, label_map,
     color_list, style_list, ticks_fs, legend_fs, lw,
     oracle= oracle, soln_list= soln_list, label_list= label_list, key = 'iou_0_7_med', multiply= 1, suffix= 'performance')
func(domain,
     height_arr, new_height_arr, color1, color2, label_map,
     color_list, style_list, ticks_fs, legend_fs, lw,
     oracle= oracle, soln_list= soln_list, label_list= label_list, key = 'iou_0_5_med', multiply= 1, suffix= 'performance')


# ==================================================================================================
# With encodings only PP versions
# ==================================================================================================
soln_list  = [gup_coord_inverted_pp, gup_npe_pp, gup_fourier_pp, run_004_pp]
color_list = [params.color_seaborn_0, params.color_seaborn_1, params.color_seaborn_15, params.color_seaborn_4, params.color_seaborn_5]
label_list = ['Coord Inverted', 'Coord NPE', 'Coord Fourier', 'Coord Vanilla', 'Baseline Ht0']
style_list = ['solid', 'solid', 'solid', 'solid', 'solid']

func(domain,
     height_arr, new_height_arr, color1, color2, label_map,
     color_list, style_list, ticks_fs, legend_fs, lw,
     oracle= oracle, soln_list= soln_list, label_list= label_list, key = 'iou_0_7_med', multiply= 1, suffix= 'performance_with_encoding')
func(domain,
     height_arr, new_height_arr, color1, color2, label_map,
     color_list, style_list, ticks_fs, legend_fs, lw,
     oracle= oracle, soln_list= soln_list, label_list= label_list, key = 'iou_0_5_med', multiply= 1, suffix= 'performance_with_encoding')

# ==================================================================================================
# With oracle PP encodings
# ==================================================================================================
soln_list  = [gup_oracle_normal, gup_oracle_normal_inv_depth, gup_oracle_inv_depth, gup_oracle_depth, gup_kprpe, gup_coord_inverted_pp]
color_list = [params.color_seaborn_1, params.color_seaborn_15, params.color_seaborn_2, params.color_seaborn_3, 'b', params.color_seaborn_5]
label_list = ['Normal', 'Normal/Depth', '1/Depth', 'Depth', 'KPRPE', 'Coord Inverted']
style_list = ['solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid']

func(domain,
     height_arr, new_height_arr, color1, color2, label_map,
     color_list, style_list, ticks_fs, legend_fs, lw,
     oracle= oracle, soln_list= soln_list, label_list= label_list, key = 'iou_0_5_med', multiply= 1, suffix= 'performance_with_oracle_encoding', ymax= 82)

# ==================================================================================================
# With oracle PP encodings
# ==================================================================================================
soln_list  = [gup_oracle_normal_all, gup_oracle_normal_fl, gup_oracle_normal, gup_coord_inverted_pp]
color_list = [params.color_seaborn_0, params.color_seaborn_1, params.color_seaborn_2, params.color_seaborn_5]
label_list = ['Oracle Norml All', 'Oracle Norml FL', 'Oracle Norml F', 'Coord Inverted']
style_list = ['dashed', 'dashed', 'dashed', 'solid', 'solid', 'solid']

func(domain,
     height_arr, new_height_arr, color1, color2, label_map,
     color_list, style_list, ticks_fs, legend_fs, lw,
     oracle= oracle, soln_list= soln_list, label_list= label_list, key = 'iou_0_5_med', multiply= 1, suffix= 'performance_with_normal_encoding', ymax= 82)


# ==================================================================================================
# With oracle PP encodings
# ==================================================================================================
soln_list  = [gup_oracle_normal_fl, gup_dsine_normal_fl, gup_metnor_normal_fl, gup_coord_inverted_pp]
color_list = [params.color_seaborn_1, params.color_seaborn_1, params.color_seaborn_3, params.color_seaborn_5]
label_list = ['Oracle Norml FL', 'DSINE Norml FL', 'Metric3D Norml FL', 'Coord Inverted']
style_list = ['dashed', 'solid', 'solid', 'solid', 'solid']

func(domain,
     height_arr, new_height_arr, color1, color2, label_map,
     color_list, style_list, ticks_fs, legend_fs, lw,
     oracle= oracle, soln_list= soln_list, label_list= label_list, key = 'iou_0_5_med', multiply= 1, suffix= 'performance_with_normal_encoding_2', ymax= 82)

# ==================================================================================================
# With Transformed Images
# ==================================================================================================
soln_list  = [gup_metnor_normal_fl, gup_dsine_normal_transf]
color_list = [params.color_seaborn_3, params.color_seaborn_5]
label_list = ['DSINE', 'DSINE Transf']
style_list = ['solid', 'solid', 'solid', 'solid']

func(domain,
     height_arr, new_height_arr, color1, color2, label_map,
     color_list, style_list, ticks_fs, legend_fs, lw,
     oracle= oracle, soln_list= soln_list, label_list= label_list, key = 'iou_0_5_med', multiply= 1, suffix= 'performance_with_normal_encoding_3', ymax= 82)

soln_list  = [gup_dsine_normal_bev, gup_dsine_normal_transf_bev]
color_list = [params.color_seaborn_3, params.color_seaborn_5]
label_list = ['DSINE', 'DSINE Transf']
style_list = ['solid', 'solid', 'solid', 'solid']

func(gup_bev,
     height_arr, new_height_arr, color1, color2, label_map,
     color_list, style_list, ticks_fs, legend_fs, lw,
     oracle= oracle_bev, soln_list= soln_list, label_list= label_list, key = 'iou_0_5_med', multiply= 1, suffix= 'performance_with_normal_encoding_3_bev', ymax= 82)

# ==================================================================================================
# With Cameras as Rays
# ==================================================================================================
soln_list  = [gup_oracle_normal_fl, gup_dsine_normal_fl, gup_directions_plucker, gup_plucker, gup_directions]
color_list = [params.color_seaborn_1, params.color_seaborn_1, params.color_seaborn_2, params.color_seaborn_3, params.color_seaborn_4]
label_list = ['Oracle Norml FL', 'DSINE Norml FL', 'Directions+Plucker', 'Plucker', 'Directions']
style_list = ['dashed', 'solid', 'solid', 'solid', 'solid']

func(domain,
     height_arr, new_height_arr, color1, color2, label_map,
     color_list, style_list, ticks_fs, legend_fs, lw,
     oracle= oracle, soln_list= soln_list, label_list= label_list, key = 'iou_0_5_med', multiply= 1, suffix= 'performance_with_rays', ymax= 82)
"""
# ==================================================================================================
# With Normals as Inputs
# ==================================================================================================
soln_list  = [gup_normal, gup_oracle_normal_fl, gup_pp]
color_list = [params.color_seaborn_0, params.color_seaborn_1, params.color_seaborn_4]
label_list = ['Normal', 'Image+Normal', 'Image PP']
style_list = ['solid', 'solid', 'solid']

func(domain,
     height_arr, new_height_arr, color1, color2, label_map,
     color_list, style_list, ticks_fs, legend_fs, lw,
     oracle= oracle, soln_list= soln_list, label_list= label_list, key = 'iou_0_5_med', multiply= 1, suffix= 'performance_with_normals', ymax= 85)