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
from lib.helpers.util import inch_2_meter

lw = params.lw + 2
fs         = 30
legend_fs  = 24
legend_fs2 = 24
ticks_fs   = fs - 4
matplotlib.rcParams.update({'font.size': fs})
color_oracle   = params.color_green
color_baseline = params.color_seaborn_4#'gold'
label_oracle = 'Oracle'
label_baseline = 'Source'
style_list = ['solid', 'solid', 'solid', 'solid', 'solid']

height_arr = inch_2_meter(np.array([-27, -24, -18, -12, -6, 0, 6, 12, 18, 24, 30]))
print(height_arr)

gup =   np.array([
              [9.46, 11.36, 16.42, 22.27, 33.34, 53.82, 41.07, 30.05, 19.87, 11.29, 7.23],
              [41.66, 46.81, 53.83, 60.12, 70.23, 76.46, 70.51, 66.81, 60.35, 52.41, 42.97],
              [0.53, 0.48, 0.43, 0.36, 0.26, 0.03, -0.18, -0.29, -0.4, -0.52, -0.63]
              ])

gup_oracle = np.array([
              [70.96, 66.77, 66.23, 63.29, 62.86, 53.82, 56.30, 59.65, 61.25, 61.26, 62.25],
              [83.88, 83.59, 83.65, 81.36, 81.11, 76.47, 78.99, 80.32, 81.83, 83.02, 83.96],
              [0.03, -0.01, 0, 0, 0.02, 0.03, 0.03, 0.03, 0.05, 0.05, 0.03]
              ])

gup_plucker = np.array([
              [8.43, 11.46, 15.58, 17.79, 30.44, 55.56, 40.05, 30.62, 21.72, 15.29, 10.13],
              [37.1, 41.64, 49.32, 52.88, 66.03, 76.57, 70.05, 63.5, 59.4, 49.67, 43.22],
              [0.55, 0.5, 0.44, 0.4, 0.29, 0.03, -0.18, -0.27, -0.39, -0.51, -0.61]
              ])

gup_unidrive = np.array([
              [10.83,11.38,17.29,20.2,38.53,53.88,42.03,29.09,17.0,8.77,5.54],
              [42.3,44.86,52.64,62.69,71.6,76.43,73.21,67.38,59.79,49.15,39.33],
              [0.51, 0.5, 0.4, 0.35, 0.23, 0.03, -0.16, -0.3, -0.43, -0.56, -0.67]
              ])

gup_ground = np.array([
              [0.98, 1.52, 3.88, 11.96, 19.76, 26.61, 25.06, 22, 15.53, 9.4, 5.39],
              [14.21, 19.49, 30.55, 41.74, 47.59, 51.97, 49.98, 48.11, 44.2, 38.48, 31.42],
              [-0.9, -0.81, -0.63, -0.4, -0.25, -0.16, -0.1, 0.04, 0.2, 0.31, 0.45]
              ])
gup_ground[2] -= gup_ground[2, 5]

gup_unidriveplusplus = np.array([
              [10.73, 11.47, 20.04, 28.62, 43.97, 53.82, 47.33, 38.54, 23.69, 17.71, 12.27],
              [47.81, 51.29, 60.68, 67.19, 71.83, 76.46, 71.24, 69.23, 63.89, 58.9, 53.08],
              [0.39, 0.39, 0.28, 0.24, 0.15, 0.03, -0.07, -0.19, -0.35, -0.4, -0.48]
              ])

# Product
gup_disparity_relu_product = np.array([
              [19.45, 26.43, 32.13, 36.92, 44.58, 55.68, 48.76, 42.91, 39.1, 33.88, 27.33],
              [53.4, 55.53, 61.27, 65.77, 71.72, 74.47, 70.94, 68.56, 68.48, 66.6, 61.98],
              [-0.07, 0, 0.05, 0.11, 0.14, 0.05, -0.04, -0.01, -0.01, -0.01, 0.02]
              ])

gup_depth_relu_product = np.array([
              [18.95, 23.11, 30.97, 34.85, 44.2, 55.14, 47.89, 41.92, 38.87, 33.66, 29.59],
              [51.69, 56.9, 64.11, 64.34, 70.67, 73.93, 70.93, 68.13, 66.46, 67.02, 64.12],
              [-0.07, -0.07, -0.02, 0.05, 0.09, 0.03, -0.05, -0.04, -0.04, -0.03, -0.01]
              ])


gup_aug_0_30 = np.array([
              [12.69, 17.02, 26.7, 33.77, 46.46, 57.91, 51.01, 39.74, 36.68, 54.59, 65.59],
              [53.23, 60.47, 67.4, 69.62, 73.26, 76.86, 77.09, 73.96, 74.71, 79.35, 86.16],
              [0.41, 0.34, 0.32, 0.28, 0.21, 0.04, -0.01, 0.19, 0.3, 0.19, 0.02]
              ])

gup_aug_0_m27 = np.array([
              [69.45, 47.48, 45.17, 47.28, 50.27, 58.18, 43.62, 34.12, 27.17, 19.63, 14.2],
              [84.78, 78.44, 77.26, 74.95, 74.11, 77.12, 73.94, 71.08, 67.84, 60.54, 54.93],
              [-0.02, 0.11, 0.18, 0.2, 0.16, 0, -0.18, -0.27, -0.34, -0.41, -0.49]
              ])

gup_m12 = np.array([
              [26.92, 30.15, 41.08, 63.29, 45.17, 34.05, 24.43, 18.03, 12.34, 8.81, 5.58],
              [61.8, 66.98, 72.75, 81.36, 73.65, 67.75, 64.18, 60.13, 54.18, 45.39, 39.19],
              [0.32, 0.27, 0.22, 0, -0.17, -0.24, -0.33, -0.41, -0.49, -0.58, -0.66]
              ])

gup_12 = np.array([
              [3.88, 3.56, 6.23, 10.85, 12.48, 16.47, 28.76, 59.65, 41.56, 21.41, 11.58],
              [26.62, 27, 35.6, 42.74, 47.82, 54.33, 66.94, 80.32, 77.22, 69.33, 58.04],
              [0.75, 0.78, 0.65, 0.56, 0.51, 0.46, 0.33, 0.03, -0.21, -0.37, -0.5]
              ])


# Teaser
soln_list  = [gup_ground, gup_disparity_relu_product]
color_list = [params.color_seaborn_2, params.color_seaborn_0]
label_list = ['Ground', 'CHARM3R']

plot_height(gup, height_arr, color_oracle, color_baseline, color_list, style_list, ticks_fs, legend_fs, lw,
            oracle= gup_oracle, soln_list= soln_list, label_list= label_list, suffix='teaser',
            label_oracle= "Oracle", label_baseline= "Source (Regress)"
            )

# Teaser for Slides
soln_list  = []
color_list = []
label_list = []

plot_height(gup, height_arr, color_oracle, color_baseline, color_list, style_list, ticks_fs, legend_fs, lw,
            oracle= gup_oracle, soln_list= soln_list, label_list= label_list, suffix='teaser_slides',
            label_oracle= "Oracle", label_baseline= "Source"
            )

# Teaser for Slides
soln_list  = [gup_ground]
color_list = [params.color_seaborn_2]
label_list = ['Ground']

plot_height(gup, height_arr, color_oracle, color_baseline, color_list, style_list, ticks_fs, legend_fs, lw,
            oracle= gup_oracle, soln_list= soln_list, label_list= label_list, suffix='teaser_sl2',
            label_oracle= "Oracle", label_baseline= "Source (Regress)"
            )

# GUP SoTA Comparison
soln_list  = [gup_disparity_relu_product, gup_unidriveplusplus, gup_unidrive, gup_plucker][::-1]
color_list = [params.color_seaborn_0, params.color_seaborn_15, params.color_seaborn_2, params.color_seaborn_3][::-1]
label_list = ['Ours', 'CHARM3R', 'UniDrive++', 'UniDrive', 'Plucker'][::-1]

plot_height(gup, height_arr, color_oracle, color_baseline, color_list, style_list, ticks_fs, legend_fs2, lw,
            oracle= gup_oracle, soln_list= soln_list, label_list= label_list, suffix='gup_sota',
            label_oracle= label_oracle, label_baseline= label_baseline)


# GUP Augmentation Comparison
soln_list  = [gup_disparity_relu_product, gup_aug_0_m27][::-1]
color_list = [params.color_seaborn_0, params.color_seaborn_2][::-1]
label_list = ['CHARM3R Train H= {0}', 'GUPNet Train H= {0,-0.70}'][::-1]

plot_height(gup, height_arr, color_oracle, color_baseline, color_list, style_list, ticks_fs, legend_fs2, lw,
            oracle= gup_oracle, soln_list= soln_list, label_list= label_list, suffix='gup_aug',
            label_oracle= label_oracle, label_baseline= 'GUPNet Train H= {0}')

# GUP Only Height0
soln_list  = [gup_12, gup_m12][::-1]
color_list = [params.color_seaborn_15, params.color_seaborn_2][::-1]
label_list = ['Train H= -0.30m', 'Train H= +0.30m'][::-1]
plot_height(gup, height_arr, color_oracle, color_baseline, color_list, style_list, ticks_fs, legend_fs2, lw,
            oracle= gup_oracle, soln_list= soln_list, label_list= label_list, suffix='gup_only_ht',
            label_oracle= label_oracle, label_baseline= 'Train H= 0m')

"""
# ==================================================================================================
# Sum
# ==================================================================================================
depth_relu_sum = np.array([
              [0.58, 1.6, 4.62, 10.9, 19.99, 45.75, 25.47, 13.58, 8.08, 3.5, 1.83],
              [11.25, 19.16, 31.95, 43.38, 59.84, 74.11, 62.94, 54.52, 45.95, 34.88, 25.71]
              ])
disparity_relu_sum = np.array([
              [0.75, 1.46, 3.27, 6.31, 11.75, 22.33, 17.29, 15.02, 14.54, 9.72, 6.47],
              [10.09, 15.77, 22.86, 32.91, 41.05, 48.5, 44.94, 39.68, 37.08, 34.31, 31.61]
              ])

soln_list  = [depth_relu_sum, disparity_relu_sum]
color_list = [params.color_seaborn_0, params.color_seaborn_15, params.color_seaborn_5]
label_list = ['Disp (Sigmoid)', 'Disp (ReLU)']

# plot_height(domain,
#             height_arr, color1, color2,
#             color_list, style_list, ticks_fs, legend_fs, lw,
#             oracle= oracle, soln_list= soln_list, label_list= label_list, suffix= 'performance_sum')


# ==================================================================================================
# Product
# ==================================================================================================
soln_list  = [gup_disparity_relu_product, monodetr]
color_list = [params.color_seaborn_15, params.color_seaborn_0]
label_list = ['Ground', 'MonoDETR']

plot_height(gup, height_arr, color_oracle, color_baseline, color_list, style_list, ticks_fs, legend_fs, lw,
            oracle= gup_oracle, soln_list= soln_list, label_list= label_list, suffix='performance_regression_ground_monodetr',
            label_oracle= label_oracle, label_baseline= label_baseline)

# ==================================================================================================
# All data
# ==================================================================================================
gup_carla_0_6 = np.array([
              [16.66, 20.7, 32.78, 43.86, 56.08, 64.05, 65.31, 65.46, 60.21, 53.82, 49.88 ],
              [53.18, 57.57, 66.26, 70.94, 77.4, 78.88, 79.64, 80.9, 81.62, 79.46, 76.77 ],
              [0.4, 0.37, 0.31, 0.23, 0.15, 0.06, 0.01, -0.06, -0.12, -0.18, -0.25]
              ])
gup_carla_0_12 = np.array([
              [32.27, 39.39, 47.59, 54.87, 61.91, 63.29, 64, 65.3, 65.22, 63.03, 59.3 ],
              [68.3, 73.94, 76.81, 77.48, 79.86, 81.36, 81.81, 82.63, 84.01, 82.61, 82.88 ],
              [0.21, 0.2, 0.17, 0.13, 0.08, 0.04, 0.01, -0.01, -0.05, -0.09, -0.13]
              ])
gup_carla_0_18 = np.array([
              [3.59, 3.9, 4.8, 6.19, 5.82, 5.86, 5.66, 5.78, 6.03, 5.86, 5.76 ],
              [38.97, 39.47, 43.17, 46.21, 44.68, 43.16, 43.53, 46.28, 46.79, 47.91, 47.91 ],
              [0.28, 0.26, 0.22, 0.18, 0.12, 0.06, 0.02, -0.01, -0.04, -0.09, -0.13]
              ])
gup_carla_0_24 = np.array([
              [3.61, 3.74, 4.97, 5.66, 6.15, 5.72, 5.63, 5.61, 5.84, 5.86, 5.92 ],
              [39.7, 41.98, 45.67, 46.23, 44.78, 43.09, 44.7, 46.19, 46.59, 47.83, 50.44 ],
              [0.22, 0.21, 0.18, 0.15, 0.09, 0.04, 0, -0.02, -0.04, -0.05, -0.09]
              ])


soln_list  = [gup_carla_0_24, gup_carla_0_18, gup_carla_0_12, gup_carla_0_6]
color_list = [params.color_seaborn_1, params.color_seaborn_0, params.color_seaborn_15, params.color_seaborn_4]
label_list = ['Ht 0+1+2+3+4', 'Ht 0+1+2+3', 'Ht 0+1+2', 'Ht 0+1']

plot_height(gup, height_arr, color_oracle, color_baseline, color_list, style_list, ticks_fs, legend_fs, lw,
            oracle= gup_oracle, soln_list= soln_list, label_list= label_list, suffix='performance_aug_all_data',
            label_oracle= label_oracle, label_baseline= 'Ht 0')


# Plucker append
gup_0_18_plucker_2 = np.array([
              [3.28, 3.55, 4.66, 4.93, 5.2, 4.86, 5.03, 5, 5.14, 4.9, 4.21],
              [32, 35.72, 41.16, 40.71, 40.38, 38.76, 38.3, 39.67, 42.27, 42.66, 40.43],
              [0, 0.13, 0.15, 0.16, 0.15, 0.07, 0.05, 0.01, -0.01, -0.1, -0.14]
              ])
gup_0_18_plucker_3 = np.array([
              [1.65, 1.75, 2.64, 3.41, 3.29, 3.05, 3.16, 3.13, 3.1, 2.96, 2.35],
              [25.78, 28.65, 31.5, 31.39, 32.65, 31.5, 31.97, 32.57, 34.64, 32.76, 32.47],
              [0.03, 0.07, 0.1, 0.08, 0.07, 0.03, 0.01, -0.04, -0.07, -0.12, -0.18]
              ])

soln_list  = [gup_carla_0_18, gup_0_18_plucker_2, gup_0_18_plucker_3]
color_list = [params.color_seaborn_0, params.color_seaborn_15, params.color_seaborn_4]
label_list = ['Ht 0+1+2+3', 'Plucker v1', 'Plucker v2']

plot_height(gup, height_arr, color_oracle, color_baseline, color_list, style_list, ticks_fs, legend_fs, lw,
            oracle= gup_oracle, soln_list= soln_list, label_list= label_list, suffix='plucker_append',
            label_oracle= label_oracle, label_baseline= label_baseline)


# New models
run_1183 = np.array([
              [1.58, 2.15, 4.62, 9.7, 19.39, 28.84, 25.32, 22.2, 15.83, 9.1, 5.85],
              [18.09, 23.11, 32.1, 41, 46.62, 55.12, 48.09, 45.82, 42.72, 38.44, 32.28],
              ])

merged = np.array([
              [13.87, 17.81, 22.93, 28.48, 39.8, 51.07, 40.26, 33.4, 26.24, 21.04, 17.59],
              [49.63, 54.02, 60.42, 64.78, 70.96, 76.44, 70.54, 67.75, 64.33, 57.63, 53.91]
              ])

soln_list  = [merged, run_1183, gup_unidrive][::-1]
color_list = [params.color_seaborn_0, params.color_seaborn_15, params.color_seaborn_4][::-1]
label_list = ['Merged', 'Ground', 'UniDrive'][::-1]

# plot_height(domain, height_arr, color1, color2, color_list, style_list, ticks_fs, legend_fs, lw,
#             oracle= oracle, soln_list= soln_list, label_list= label_list, suffix= 'performance_merged')
"""

# ==================================================================================================
# DEVIANT
# ==================================================================================================
dev =   np.array([
              [8.63, 11.35, 17.43, 21.8, 34.47, 50.18, 36.45, 24.76, 17.07, 9.95, 6.25],
              [40.24, 45.01, 55.51, 60.11, 67.48, 73.78, 67.21, 63.97, 57.57, 47.85, 41.74],
              [0.46, 0.42, 0.36, 0.3, 0.22, 0.01, -0.18, -0.3, -0.43, -0.55, -0.65]
              ])

dev_oracle = np.array([
              [71.97, 63.81, 66.11, 63.97, 54.14, 50.18, 54.14, 59.74, 60.46, 60.87, 62.56],
              [84.56, 83.78, 84.15, 81.82, 76.51, 73.78, 76.52, 80.77, 81.74, 82.38, 83.94],
              [0, 0.02, -0.01, 0.02, 0.01, 0.01, 0.04, 0.05, 0.06, 0.03, 0.02]
              ])

dev_plucker = np.array([
              [8.43, 11.46, 16.02, 17.79, 30.44, 51.32, 40.05, 30.62, 16.72, 15.29,9.52],
              [38.24, 41.64, 52.32, 52.88, 66.03, 73.91, 70.05, 63.5, 56.57, 49.67, 44.22],
              [0.55, 0.5, 0.44, 0.4, 0.29, 0.03, -0.18, -0.27, -0.39, -0.51, -0.61]
              ])

dev_unidrive = np.array([
              [8.33, 11.54, 17.05, 23.84, 36.19, 50.18, 39.51, 30.41, 19.11, 9.85, 6.56],
              [41.4, 47.22, 55.46, 63.16, 68, 73.78, 69.56, 65.08, 59.23, 47.33, 41.27],
              [0.46, 0.42, 0.34, 0.28, 0.18, 0.01, -0.13, -0.25, -0.4, -0.56, -0.64]
              ])

dev_unidriveplusplus = np.array([
              [6.73, 10.85, 18.14, 25.47, 37.16, 50.18, 41.06, 32.39, 20.8, 16.07, 12.03],
              [42.91, 49.97, 59.38, 64.04, 70.19, 73.78, 68.18, 67.65, 62.08, 55.81, 52.36],
              [0.37, 0.28, 0.21, 0.2, 0.14, 0.01, -0.12, -0.24, -0.38, -0.43, -0.47]
              ])

# Product
dev_disparity_relu_product = np.array([
              [17.11,18.83,24.81,31.77,39.85,48.74,43.25,39.37,33.9,30.79,26.24],
              [49.28,52.45,56.44,61.73,67.21,70.21,67.41,65.29,66.15,66.33,63.6],
              [0.01,0.04,0.05,0.06,0.11,0.03,-0.06,-0.05,-0.05,-0.05,-0.02]
              ])

soln_list  = [dev_disparity_relu_product, dev_unidriveplusplus, dev_unidrive, dev_plucker][::-1]
color_list = [params.color_seaborn_0, params.color_seaborn_15, params.color_seaborn_2, params.color_seaborn_3][::-1]
label_list = ['CHARM3R', 'UniDrive++', 'UniDrive', 'Plucker'][::-1]

plot_height(dev, height_arr, color_oracle, color_baseline, color_list, style_list, ticks_fs, legend_fs2, lw,
            oracle= dev_oracle, soln_list= soln_list, label_list= label_list, suffix='dev_sota',
            label_oracle= label_oracle, label_baseline= label_baseline)

# ==================================================================================================
# MonoDETR
# ==================================================================================================
monodetr = np.array([
              [0.01, 0.02, 0.12, 0.71, 3.98, 9.81, 13.63, 10.46, 6.81, 4.27, 2.61],
              [0.53, 1.6, 6.2, 14.05, 25.54, 33.55, 38.88, 41.88, 36.54, 28.09, 21.16],
              [2.13, 1.88, 1.55, 1.22, 0.86, 0.55, 0.22, -0.05, -0.27, -0.51, -0.72],
            ])

