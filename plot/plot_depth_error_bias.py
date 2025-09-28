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

# ==================================================================================================
# Bias data Mean/Median Error on Boxes
# ==================================================================================================
gup_mean      = np.array([0.00, -0.292, -0.498, -0.763, -1.104, -1.487])
gup_median    = np.array([0.00, -0.250, -0.420, -0.670, -0.983, -1.380])
gup_pp_mean   = np.array([0.00, 0.040, -0.078, -0.096, -0.114, -0.054])
gup_pp_median = np.array([0.00, 0.000, 0.000, 0.000, 0.000, 0.050])

gup_coord_mean   = np.array([0.00, -0.292, -0.47, -0.66, -0.82, -0.95])
gup_coord_median = np.array([0.00, -0.243, -0.39, -0.58, -0.73, -0.86])

oracle = np.zeros((6,))

color_list = [params.color_green, 'gold', params.color_seaborn_15]
soln_list  = [oracle, gup_mean, gup_coord_mean]
label_list = ['W/o gap', 'Vanilla', '+Coord']

height_arr = np.array([0, 6, 12, 18, 24, 30]) * 0.0254
height_arr = np.round(height_arr, 2)
new_height_arr = []
for i, h in enumerate(height_arr):
    new_height_arr.append(str(h) + "\n" + str(i))

# Do linear regression
# from sklearn.linear_model import LinearRegression
# x = height_arr.reshape((-1, 1))
# y = gup_bias_median
# model = LinearRegression().fit(x, y)
# r_sq = model.score(x, y)
# print(f"coefficient of determination: {r_sq}")
# print(f"intercept: {model.intercept_}")
# print(f"slope: {model.coef_}")

plt.figure(figsize= params.size, dpi= params.DPI)
plt.plot(height_arr, gup_mean, label='Vanilla Mean', lw= params.lw, c= params.color_seaborn_3)
plt.plot(height_arr, gup_median, label='Vanilla Median', lw= params.lw, c= params.color_seaborn_1)
plt.plot(height_arr, gup_pp_mean, label='Van+ PP Mean', lw= params.lw, c= params.color_seaborn_3, linestyle='--')
plt.plot(height_arr, gup_pp_median, label='Van+ PP Median', lw= params.lw, c= params.color_seaborn_1, linestyle='--')
plt.xlabel(r'$\Delta$' + 'Height (m)')
plt.ylabel('Depth Err' + r' $(Pred-GT)$' + '(m)')
plt.xlim(left= -0.02)
plt.ylim(-2.2, 2.2)
plt.grid(True)
plt.xticks(height_arr, labels=new_height_arr)
plt.legend(loc='upper right', borderaxespad= params.legend_border_axes_plot_border_pad, borderpad= params.legend_border_pad, labelspacing= params.legend_vertical_label_spacing, handletextpad= params.legend_marker_text_spacing)
savefig(plt, "images/depth_error_bias_teaser.png")
plt.close()

assert len(soln_list) <= len(color_list)
assert len(label_list) <= len(color_list)

plt.figure(figsize= params.size, dpi= params.DPI)
for soln, label, color in zip(soln_list, label_list, color_list):
    plt.plot(height_arr, soln, label= label, lw= params.lw, c= color)
plt.xlabel(r'$\Delta$' + 'Height (m)')
plt.ylabel('Depth Err' + r' $(Pred-GT)$' + '(m)')
plt.xlim(left= -0.02)
plt.ylim(-2.2, 2.2)
plt.grid(True)
plt.xticks(height_arr, labels=new_height_arr)
plt.legend(loc='upper right', borderaxespad= params.legend_border_axes_plot_border_pad, borderpad= params.legend_border_pad, labelspacing= params.legend_vertical_label_spacing, handletextpad= params.legend_marker_text_spacing)
savefig(plt, "images/depth_error_bias.png")
plt.close()




# plt.figure(figsize= params.size, dpi= params.DPI)
# plt.plot(height_arr, y2_arr, label= 'Mean'  , lw= params.lw, c= params.color_seaborn_1)
# plt.plot(height_arr, z2_arr, label= 'Median', lw= params.lw, c= params.color_seaborn_3)
# plt.xlabel(r'$\Delta$' + 'Height (m)')
# plt.ylabel('Abs Depth Error (m)')
# plt.xlim(left= -0.02)
# plt.ylim(-0.2, 2.2)
# plt.grid(True)
# plt.xticks(height_arr)
# plt.legend(loc='lower right', borderaxespad= params.legend_border_axes_plot_border_pad, borderpad= params.legend_border_pad, labelspacing= params.legend_vertical_label_spacing, handletextpad= params.legend_marker_text_spacing)
# savefig(plt, "images/depth_error_abs.png")
# plt.close()