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
import argparse

np.set_printoptions   (precision= 2, suppress= True)
torch.set_printoptions(precision= 2, sci_mode= False)

from plot.common_operations import *
import plot.plotting_params as params
import matplotlib

from lib.helpers.file_io import read_lines

def parse(cnt):
    w = cnt // (num_formulations * num_ds * num_conv_layers)
    remain = cnt % (num_formulations * num_ds * num_conv_layers)
    f = remain // (num_ds * num_conv_layers)
    remain = remain % (num_ds * num_conv_layers)
    d = remain // (num_conv_layers)
    l = cnt % (num_conv_layers)

    return w, f, d, l

# ==================================================================================================
# Main Starts here
# ==================================================================================================
parser = argparse.ArgumentParser(description='plot')
parser.add_argument('--file' , type=str, default = "test/ee_depth_reciprocal_error_carla_max.log", help='file to evaluate')
args = parser.parse_args()

file_path = args.file

num_warp = 2         # True, False
num_formulations = 2 # disparity, depth
num_ds = 3           # 4, 8, 16
num_conv_layers = 4  # 1, 2, 3, 4

our = np.zeros((num_warp, num_formulations, num_ds, num_conv_layers))
cnn = np.zeros_like(our)

lines = read_lines(file_path, strip= True)
cnn_cnt = 0
our_cnt = 0
for curr_line in lines:
    if "EE Model0" in curr_line:
        w, f, d, l = parse(cnn_cnt)
        cnn[w, f, d, l] = str(curr_line.split(" ")[-1])
        cnn_cnt += 1
    elif "EE Model1" in curr_line:
        w, f, d, l = parse(our_cnt)
        our[w, f, d, l] = str(curr_line.split(" ")[-1])
        our_cnt += 1
    else:
        pass

# Now plot
lw = params.lw
legend_fs = params.legend_fs + 2
legend_border_axes_plot_border_pad = params.legend_border_axes_plot_border_pad
legend_border_pad = params.legend_border_pad
legend_vertical_label_spacing = params.legend_vertical_label_spacing
legend_marker_text_spacing = params.legend_marker_text_spacing
c1 = params.color_seaborn_2#'gold'
c2 = params.color_seaborn_1
c11 = c1
c12 = c2
linestyles = ['solid', 'dashed', 'dotted']
conv_layers = np.arange(num_conv_layers) + 1
if "_max" in file_path:
    ymin, ymax = 0.04, 0.22
else:
    ymin, ymax  = 0.04, 0.22
if "unmasked" in file_path:
    loc = "upper right"
else:
    loc = "lower left"

for w in [0, 1]:
    plt.figure(figsize= (20, 6), dpi= params.DPI)
    plt.subplot(1,2,1)
    plt.title('Depth')
    plt.plot(conv_layers, cnn[w, 1, 0], c= c11, label='CNN 4', linestyle= linestyles[0], lw= lw)
    plt.plot(conv_layers, our[w, 1, 0], c= c12, label='Our 4', linestyle= linestyles[0], lw= lw)
    plt.plot(conv_layers, cnn[w, 1, 1], c= c11, label='CNN 8', linestyle= linestyles[1], lw= lw)
    plt.plot(conv_layers, our[w, 1, 1], c= c12, label='Our 8', linestyle= linestyles[1], lw= lw)
    plt.plot(conv_layers, cnn[w, 1, 2], c= c11, label='CNN 16', linestyle= linestyles[2], lw= lw)
    plt.plot(conv_layers, our[w, 1, 2], c= c12, label='Our 16', linestyle= linestyles[2], lw= lw)
    plt.xlim([0.9, 4.1])
    plt.ylim([ymin, ymax])
    plt.grid()
    plt.legend(loc= loc, fontsize= legend_fs, borderaxespad= legend_border_axes_plot_border_pad, borderpad= legend_border_pad, labelspacing= legend_vertical_label_spacing, handletextpad= legend_marker_text_spacing)
    plt.xlabel('#Num Conv Layers')
    plt.ylabel('Eqv Error ' + r'$(\downarrow)$')

    plt.subplot(1,2,2)
    plt.title('Disparity')
    plt.plot(conv_layers, cnn[w, 0, 0], c=c1, label='CNN 4', linestyle=linestyles[0], lw=lw)
    plt.plot(conv_layers, our[w, 0, 0], c=c2, label='Our 4', linestyle=linestyles[0], lw=lw)
    plt.plot(conv_layers, cnn[w, 0, 1], c=c1, label='CNN 8', linestyle=linestyles[1], lw=lw)
    plt.plot(conv_layers, our[w, 0, 1], c=c2, label='Our 8', linestyle=linestyles[1], lw=lw)
    plt.plot(conv_layers, cnn[w, 0, 2], c=c1, label='CNN 16', linestyle=linestyles[2], lw=lw)
    plt.plot(conv_layers, our[w, 0, 2], c=c2, label='Our 16', linestyle=linestyles[2], lw=lw)
    plt.xlim([0.9, 4.1])
    plt.ylim([ymin, ymax])
    plt.grid()
    plt.legend(loc=loc, fontsize=legend_fs, borderaxespad=legend_border_axes_plot_border_pad,
               borderpad=legend_border_pad, labelspacing=legend_vertical_label_spacing,
               handletextpad=legend_marker_text_spacing)
    plt.xlabel('#Num Conv Layers')

    path = "images/" + osp.basename(file_path).replace(".log", "")
    if w == 0:
        path += "_warped"
    savefig(plt, path + ".png")
    # plt.show()
    plt.close()