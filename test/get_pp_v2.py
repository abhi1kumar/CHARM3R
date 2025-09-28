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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

from lib.helpers.file_io import read_numpy

def err_metric(gt, prediction):
    output = r2_score(gt, prediction)
    output = np.mean(np.abs(prediction-gt))
    return output

def normalize(features, min= None, max= None):
    if min is None:
        min = np.zeros(features.shape[1])
        # min = np.min(features, axis= 0)
    if max is None:
        max = np.ones(features.shape[1])
        # max = np.max(features, axis= 0)
    feat_norm = (features - min[np.newaxis, :])/(max[np.newaxis, :] - min[np.newaxis, :])
    return feat_norm, min, max

def fit_model(features, output, val_features= None, val_output= None, output_str= None):
    feat_norm, min, max = normalize(features)
    model = LinearRegression().fit(feat_norm, output)
    print("============{}===========".format(output_str))
    print(f"slope    : {model.coef_}")
    print("Intercept: {:.2f}".format(model.intercept_))
    val_feat_norm, _, _ = normalize(val_features, min, max)
    metric = err_metric(val_output, model.predict(val_feat_norm))
    print("MAE      : {:.2f}".format(metric))

def quadratic_func(features, a, b, c):
    depth_pred, height = features
    depth_out = depth_pred/(a + b*height)  + c
    return depth_out

def fit_non_linear_model(features, output, val_features= None, val_output= None, output_str= None):
    # Fit the model using curve_fit
    p0 = 1.0, 0.13, 0.0
    popt, pcov = curve_fit(quadratic_func, (features[:, 0], features[:, 1]), output, p0= p0)
    print("============{}===========".format(output_str))
    print("Slope    :", popt[:-1])
    print("Intercept: {:.2f}".format(popt[-1]))
    metric = err_metric(val_output, quadratic_func((val_features[:, 0], val_features[:, 1]), popt[0], popt[1], popt[2]))
    print("MAE      : {:.2f}".format(metric*1.))

folder = "output/gup_carla/result_carla_wo_pp"
rel_folders = ["height0", "height6", "height12", "height18", "height24", "height30"]
height_arr = np.array([0, 6, 12, 18, 24, 30]) * 0.0254
# rel_folders = ["height6", "height12", "height18", "height24", "height30"]
# height_arr = np.array([6, 12, 18, 24, 30]) * 0.0254
complete_data = None
EPS = 1e-2

for i, (rf, h) in enumerate(zip(rel_folders, height_arr)):
    numpy_path = os.path.join(folder, rf, "pred_gt_box.npy")
    data_temp  = read_numpy(numpy_path)
    depth_pred = data_temp[:, 0]
    depth_gt   = data_temp[:, 1]
    data       = np.zeros((data_temp.shape[0], 3))
    data[:, 0] = -h
    data[:, 1] = depth_pred
    data[:, 2] = depth_gt
    if complete_data is None:
        complete_data = data
    else:
        complete_data = np.vstack((complete_data, data))
    print(np.mean(depth_pred-depth_gt))


# mask = complete_data[:, 1] < 0
# print(np.sum(mask))
np.random.seed(0)

# Split into train and val
num_inputs  = complete_data.shape[0]
all_index   = np.arange(num_inputs)
train_index = np.random.choice(num_inputs, int(0.8*num_inputs), replace=False)
val_index   = np.setdiff1d(all_index, train_index)

# print(num_inputs)
# print(np.intersect1d(train_index, val_index).shape[0])
# print(np.union1d(train_index, val_index).shape[0])

# Data for features
height     = complete_data[:, 0]
depth_pred = complete_data[:, 1]
depth_gt   = complete_data[:, 2]
depth_pred_squared = depth_pred*depth_pred
num_points = complete_data.shape[0]

print(err_metric(depth_gt[val_index], depth_pred[val_index]))
# Now construct features
features   = np.zeros((num_points, 2))
features[:, 0] = depth_pred
features[:, 1] = height
output     = depth_gt
fit_model(features[train_index], output[train_index], features[val_index], output[val_index], "Mod1 Input: Z H Output: Z_gt")



features   = np.zeros((num_points, 3))
features[:, 0] = depth_pred
features[:, 1] = height
features[:, 2] = height/depth_pred
output         = depth_gt
fit_model(features[train_index], output[train_index], features[val_index], output[val_index], "Mod2 Input: Z H H/Z Output: Z_gt")

features   = np.zeros((num_points, 4))
features[:, 0] = depth_pred
features[:, 1] = height
features[:, 2] = height/depth_pred
features[:, 3] = depth_pred/(height + EPS)
output     = depth_gt
fit_model(features[train_index], output[train_index], features[val_index], output[val_index], "Mod3 Input: Z H H/Z Z/H (Output: Z_gt")

features   = np.zeros((num_points, 5))
features[:, 0] = depth_pred
features[:, 1] = height
features[:, 2] = height/depth_pred
features[:, 3] = depth_pred/(height + EPS)
features[:, 4] = height * depth_pred
output     = depth_gt
fit_model(features[train_index], output[train_index], features[val_index], output[val_index], "Mod4 Input: Z H H/Z Z/H H*Z Output: Z_gt")

features   = np.zeros((num_points, 9))
features[:, 0] = depth_pred
features[:, 1] = height
features[:, 2] = height/depth_pred
features[:, 3] = depth_pred/(height + EPS)
features[:, 4] = height * depth_pred
features[:, 5] = depth_pred_squared
features[:, 6] = height/depth_pred_squared
features[:, 7] = depth_pred_squared/(height + EPS)
features[:, 8] = height * depth_pred_squared
output     = depth_gt
fit_model(features[train_index], output[train_index], features[val_index], output[val_index], "Mod5 Input: Z H H/Z Z/H H*Z Z2 H/Z2 Z2/H H*Z2 Output: Z_gt")

features   = np.zeros((num_points, 5))
features[:, 0] = depth_pred
features[:, 1] = height
features[:, 2] = height/depth_pred
features[:, 3] = height*height/depth_pred_squared
features[:, 4] = height * height * height/ (depth_pred * depth_pred_squared)
output     = depth_gt
fit_model(features[train_index], output[train_index], features[val_index], output[val_index], "Mod6 Input: Z H H/Z H2/Z2 H3/Z3 Output: Z_gt")

features   = np.zeros((num_points, 5))
features[:, 0] = depth_pred
features[:, 1] = height
features[:, 2] = height/depth_pred
features[:, 3] = height/depth_pred_squared
features[:, 4] = height/(depth_pred * depth_pred_squared)
output     = depth_gt
fit_model(features[train_index], output[train_index], features[val_index], output[val_index], "Mod7 Input: Z H H/Z H/Z2 H/Z3 Output: Z_gt")

features   = np.zeros((num_points, 5))
features[:, 0] = depth_pred
features[:, 1] = height
features[:, 2] = height*depth_pred
features[:, 3] = height*depth_pred_squared
features[:, 4] = height*(depth_pred * depth_pred_squared)
output     = depth_gt
fit_model(features[train_index], output[train_index], features[val_index], output[val_index], "Mod8 Input: Z H H/Z H/Z2 H/Z3 Output: Z_gt")


features   = np.zeros((num_points, 2))
features[:, 0] = depth_pred
features[:, 1] = height
output     = depth_gt
fit_non_linear_model(features[train_index], output[train_index], features[val_index], output[val_index], "Mod9 Input: Z H Output: Z_gt")

features   = np.zeros((num_points, 5))
features[:, 0] = height * depth_pred
features[:, 1] = depth_pred
features[:, 2] = height
features[:, 3] = height / depth_pred
features[:, 4] = 1.0    / depth_pred
output         = 1.0    / depth_gt
fit_model(features[train_index], output[train_index], features[val_index], output[val_index], "Mod10 Input: H*Z Z H H/Z 1/Z Output: 1/Z_gt")


features   = np.zeros((num_points, 7))
features[:, 0] = height * depth_pred
features[:, 1] = depth_pred
features[:, 2] = height
features[:, 3] = height / depth_pred
features[:, 4] = 1.0    / depth_pred
features[:, 5] = 1.0    / (height * depth_pred + EPS)
features[:, 6] = depth_pred / (height + EPS)
output         = 1.0    / depth_gt
fit_model(features[train_index], output[train_index], features[val_index], output[val_index], "Mod11 Input: H*Z Z H H/Z 1/Z 1/HZ Z/H Output: 1/Z_gt")





