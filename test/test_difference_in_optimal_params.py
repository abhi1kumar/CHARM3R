"""
    Sample Run:
    python .py
"""
import os, sys

import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

import numpy as np
import torch
np.set_printoptions   (precision= 2, suppress= True)
torch.set_printoptions(precision= 2, sci_mode= False)

from plot.common_operations import *
import plot.plotting_params as params
import matplotlib

EPS = 1e-2

def norm(input):
    a = torch.linalg.norm(input.flatten()).cpu().item()
    # a = torch.sum(torch.abs(input)).cpu().item()
    return a

def dist(inp1, inp2):
    nbline    = norm(inp1)
    noracle   = norm(inp2)
    diff      = norm(inp1-inp2)
    rel_diff  = diff/(noracle  + EPS)
    ndiff     = np.abs(nbline-noracle)
    rel_ndiff = ndiff/(noracle + EPS)
    params    = inp1.numel()
    return noracle, nbline, rel_diff, ndiff, rel_ndiff, params

model_bline  = "output/gup_carla/checkpoints/checkpoint_epoch_140.pth"
model_no_gap = "output/gup_carla_height0_6_25k/checkpoints/checkpoint_epoch_140.pth"
device       = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
plot_style   = "vanilla"
rotation     = 270
xlabel_font  = 20
ha= 'center'#'right'

weights_bline  = torch.load(model_bline, map_location= device)['model_state']
weights_oracle = torch.load(model_no_gap, map_location= device)['model_state']

keys_list   = list(weights_bline.keys())
all_data    = np.zeros((len(keys_list), 6)).astype(np.float64)

cnt = 0
x_labels = []
for k in keys_list:
    if 'num_batches_tracked' in k or 'running_var' in k:
        continue
    # print(k)
    noracle, nbline, rel_diff, ndiff, rel_ndiff, num_params = dist(weights_bline [k], weights_oracle[k])
    if rel_ndiff >= 0.1:
        print("Index= {:3d} Key = {:50s} rel_ndiff={:.2f} ndiff= {:.2f} rel_diff= {:.2f} nbline= {:.2f} noracle= {:.2f} ".format(cnt, k, rel_ndiff, ndiff, rel_diff, nbline, noracle))

        all_data[cnt, 0] = noracle
        all_data[cnt, 1] = nbline
        all_data[cnt, 2] = rel_diff
        all_data[cnt, 3] = ndiff
        all_data[cnt, 4] = rel_ndiff
        all_data[cnt, 5] = num_params

        k = k.replace("running_", "r_").replace("conv", "c").replace("backbone", "b").replace("level", "l")
        x_labels.append(k)
        cnt += 1

x = np.arange(cnt)
y = all_data[:cnt, 4]
ylabel = "Rel Difference (%)"
if plot_style == "log":
    print("Using log style...")
    y = np.log10(y)
    ylabel = "Log10 Diff"

y_labels = np.arange(0, 1.01, 0.2)
y_labels_show = np.arange(0, 101, 20)
plt.figure(figsize= (24, 6), dpi= params.DPI)
plt.scatter(x, y, label='Bline-Oracle', s= params.ms*10, c= params.color_seaborn_0)
plt.xlabel('Layer Name')
plt.xticks(ticks= x, labels= x_labels, rotation= rotation, fontsize= xlabel_font, ha= ha)
plt.yticks(ticks= y_labels, labels= y_labels_show)
plt.ylabel(ylabel)
plt.ylim(0, 1.0)
plt.legend(loc= 'upper left')
plt.grid(True)
savefig(plt, "images/difference_in_optimal_weight.png")
plt.close()

cnt = 0
x_labels = []
for k in keys_list:
    if 'num_batches_tracked' in k or 'running_var' in k:
        continue
    noracle, nbline, rel_diff, ndiff, rel_ndiff, num_params = dist(weights_bline [k], weights_oracle[k])
    if ndiff >= 1:
        print("Index= {:3d} Key = {:50s} rel_ndiff={:.2f} ndiff= {:.2f} rel_diff= {:.2f} nbline= {:.2f} noracle= {:.2f} ".format(cnt, k, rel_ndiff, ndiff, rel_diff, nbline, noracle))

        all_data[cnt, 0] = noracle
        all_data[cnt, 1] = nbline
        all_data[cnt, 2] = rel_diff
        all_data[cnt, 3] = ndiff
        all_data[cnt, 4] = rel_ndiff
        all_data[cnt, 5] = num_params

        k = k.replace("running_", "r_").replace("conv", "c").replace("backbone", "b").replace("level", "l")
        x_labels.append(k)
        cnt += 1

x = np.arange(cnt)
y = all_data[:cnt, 3]
ylabel = "Abs Difference"

plt.figure(figsize= (24, 6), dpi= params.DPI)
plt.scatter(x, y, label='Bline-Oracle', s= params.ms*20, c= params.color_seaborn_1)
plt.xlabel('Layer Name')
plt.ylabel(ylabel)
plt.xticks(ticks= x, labels= x_labels, rotation= rotation, fontsize= xlabel_font, ha= ha)
# plt.yticks(ticks= y_labels, labels= y_labels_show)
plt.ylim(0, np.ceil(np.max(y)))
plt.legend(loc= 'upper left')
plt.grid(True)
savefig(plt, "images/difference_in_optimal_weight_absdiff.png")
plt.close()