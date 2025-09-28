"""
    Sample Run:
    python test/get_oracle_with_ground_depth.py
"""
import os, sys
sys.path.append(os.getcwd())

import os.path as osp
import glob
import numpy as np
import torch
import torch.nn as nn
import random
import copy

np.set_printoptions   (precision= 2, suppress= True)
torch.set_printoptions(precision= 2, sci_mode= False)

from plot.common_operations import *
import plot.plotting_params as params
import matplotlib
import matplotlib.pyplot as plt

from lib.helpers.ground_plane import ground_depth, inverse_ground
from lib.datasets.kitti_utils import get_calib_from_file
from lib.helpers.file_io import read_csv, read_lines, write_lines, read_image, save_numpy
from lib.helpers.rpn_util import evaluate_kitti_results_verbose, get_MAE
from lib.helpers.math_3d import project_3d_points_in_4D_format

def array_to_space_separated_string(array):
    """Converts a NumPy array of objects to a space-separated string, preserving lines.

    Args:
        array: The NumPy array to convert.

    Returns:
        A string representing the array, with elements separated by spaces and lines preserved.
    """
    # Convert each element to a string and join with spaces
    lines = []
    for row in array:
        line = ""
        for i, element in enumerate(row):
            if i == 1 or i >= 3:
                line += str("{:.2f}".format(element))
            else:
                line += str(element)

            if i != len(row) - 1:
                line += " "

        lines.append(line)

    # Join the lines with newlines
    t = "\n".join(lines)
    return t + "\n"

def init_torch(rng_seed, cuda_seed):
    """
    Initializes the seeds for ALL potential randomness, including torch, numpy, and random packages.

    Args:
        rng_seed (int): the shared random seed to use for numpy and random
        cuda_seed (int): the random seed to use for pytorch's torch.cuda.manual_seed_all function
    """
    # seed everything
    os.environ['PYTHONHASHSEED'] = str(rng_seed)
    torch.manual_seed(rng_seed)
    np.random.seed(rng_seed)
    random.seed(rng_seed)
    torch.cuda.manual_seed(cuda_seed)
    torch.cuda.manual_seed_all(cuda_seed)

    # make the code deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==================================================================================================
# Main starts here
# ==================================================================================================
init_torch(0, 0)
ext_config    = "height30"
data_folder   = osp.join("data/carla", ext_config, "validation")
# data_folder   = osp.join("data/KITTI", "training")
data_folder   = osp.join("data/coda", "training")
# data_folder   = osp.join("data/waymo", "validation")
output_folder = "output/oracle_ground/data"
use_3d_center = True
use_float_gd  = True
downsample    = 1

if "KITTI" in data_folder:
    dataset        = "KITTI"
    LBL_FOLDER_REL = "label_2"
    IMG_FOLDER_REL = "image_2"
    IMG_EXT        = ".png"
    alpha          = 0.0
    subtract_one   = False
    npy_file_path  = osp.join(output_folder.replace("/data", ""), dataset + "_all_data.npy")
elif "coda" in data_folder:
    dataset = "Coda"
    LBL_FOLDER_REL = "label_2"
    IMG_FOLDER_REL = "image_2"
    IMG_EXT = ".jpg"
    alpha = 0.0
    subtract_one = False
    npy_file_path = osp.join(output_folder.replace("/data", ""), dataset + "_all_data.npy")
elif "waymo" in data_folder:
    dataset        = "Waymo"
    LBL_FOLDER_REL = "label"
    IMG_FOLDER_REL = "image"
    IMG_EXT        = ".png"
    alpha          = 0.0
    subtract_one   = False
    npy_file_path  = osp.join(output_folder.replace("/data", ""), dataset + "_all_data.npy")
else:
    dataset        = "CARLA"
    LBL_FOLDER_REL = "label"
    IMG_FOLDER_REL = "image"
    IMG_EXT        = ".jpg"
    alpha          = 1.0
    subtract_one   = True
    npy_file_path  = osp.join(output_folder.replace("/data", ""), dataset + "_" + ext_config + "_all_data.npy")

gt_folder     = osp.join(data_folder, LBL_FOLDER_REL)
calib_files   = sorted(glob.glob(data_folder + "/calib/*.txt"))
if dataset.lower() == "coda":
    num_files     = len(calib_files)
else:
    num_files     = 1000#len(calib_files)
calib_files   = sorted(random.sample(calib_files, num_files))

print("===========================================================================================")
print("Dataset   = {}".format(dataset))
print("Downsample= {}".format(downsample))
if use_3d_center:
    print("Using projected 3D center...")
if use_float_gd:
    print("Using floating ground...")
if subtract_one:
    print("Subtracting one in bottom y...")
print("===========================================================================================")

if os.path.exists(output_folder):
    os.system("rm -rf " + output_folder)
os.makedirs(output_folder, exist_ok=True)

all_data = None

for i, calib_file in enumerate(calib_files):
    if (i+1) % 250 == 0 or (i+1) == num_files:
        print("{} Images done...".format((i+1)))

    key        = os.path.basename(calib_file).replace(".txt", "")

    calib      = get_calib_from_file(calib_file)
    p2_np      = np.concatenate((np.array(calib['P2']), np.array([[0, 0, 0, 1.]])), axis= 0)
    p2         = torch.from_numpy(p2_np).unsqueeze(0).cuda()                    # B x 4 x 4
    intrinsics = torch.from_numpy(calib["intrinsics"]).unsqueeze(0).cuda()      # B x 3 x 3
    intrins_4  = torch.eye(4).unsqueeze(0).cuda()
    intrins_4[:,:3,: 3] = intrinsics
    extrinsics = torch.linalg.inv(intrins_4) @ p2.float()                       # B x 4 x 4
    extrinsics = extrinsics[:, :3, :]                                           # B x 3 x 4
    trans      = torch.eye(3).unsqueeze(0).cuda()                               # B x 3 x 3
    cam_height = torch.from_numpy(calib["gd_to_cam"])[1, 3].unsqueeze(0).cuda() # B
    if "coda" in dataset.lower():
        cam_height -= 0.9

    image_file = osp.join(data_folder, IMG_FOLDER_REL, key + IMG_EXT)
    image      = read_image(image_file)
    im_h, im_w, _ = image.shape

    label_file = osp.join(data_folder, LBL_FOLDER_REL, key + ".txt")
    labels     = read_csv(label_file, use_pandas= True)
    output_label_file = osp.join(output_folder, key + ".txt")

    if labels is None or len(labels) <= 0:
        write_lines(output_label_file, lines_with_return_character= "")
        continue

    #        0   1    2     3   4   5  6    7    8    9    10   11   12   13    14
    # (cls, -1, -1, alpha, x1, y1, x2, y2, h3d, w3d, l3d, x3d, y3d, z3d, ry3d, score)
    categories = labels[:, 0]
    boxes      = np.array(labels[:, 1:])
    car_index  = categories == 'Car'
    if car_index is None or np.sum(car_index) <= 0:
        write_lines(output_label_file, lines_with_return_character="")
        continue
    categories = categories[car_index]
    boxes      = boxes[car_index]

    area       = (boxes[:, 6] - boxes[:, 4])*(boxes[:, 5] - boxes[:, 3])
    val_ar_ind = area > 10
    if val_ar_ind is None or np.sum(val_ar_ind) <= 0:
        write_lines(output_label_file, lines_with_return_character="")
        continue
    categories = categories[val_ar_ind]
    boxes      = boxes[val_ar_ind]

    val_dep_ind= boxes[:, 12] > -2.0
    if val_dep_ind is None or np.sum(val_dep_ind) <= 0:
        write_lines(output_label_file, lines_with_return_character="")
        continue
    categories = categories[val_dep_ind]
    boxes      = boxes[val_dep_ind]

    categories = categories[:, np.newaxis]
    num_boxes  = boxes.shape[0]
    SCORES     = np.ones((num_boxes, 1))
    boxes      = np.concatenate((boxes, SCORES), axis= 1)
    boxes_new  = copy.deepcopy(boxes)
    image_data = copy.deepcopy(boxes)

    # 2D properties
    boxes      = boxes.astype(np.float32)
    h2d        = boxes[:, 6] - boxes[:, 4]
    u          = 0.5 * (boxes[:, 5] + boxes[:, 3])
    v          = 0.5 * (boxes[:, 6] + boxes[:, 4])

    # 3D properties
    depth_old  = boxes[:, 12]
    h3d        = boxes[:, 7]
    XYZ        = boxes[:, 10:13] # N x 3
    XYZ[:, 1] -= h3d/2.0

    if use_3d_center:
        proj = project_3d_points_in_4D_format(p2_np, XYZ.T, pad_ones= True).T # N x 4
        cx   = proj[:, 0]
        cy   = proj[:, 1]
        bx   = cx
        by   = cy + 0.5*h2d + alpha*(cy - v)
    else:
        cx = u
        cy = v
        bx = cx
        by = cy + 0.5*h2d

    if subtract_one:
        by -= 1

    bx = np.clip(bx, 0.0, im_w - 1.0)
    by = np.clip(by, 0.0, im_h - 1.0)
    bx /= downsample
    by /= downsample
    U = torch.from_numpy(bx).unsqueeze(0).unsqueeze(0).cuda()  # 1 x 1 x m
    V = torch.from_numpy(by).unsqueeze(0).unsqueeze(0).cuda() # 1 x 1 x m
    depth_center          = ground_depth  (intrinsics, extrinsics, cam_height, im_h, im_w, downsample, trans, U= U, V= V)[0, 0].cpu().numpy() # m
    by_ideal              = inverse_ground(intrinsics, extrinsics, cam_height, im_h, im_w, downsample, trans, U= U, gdepth= torch.from_numpy(depth_old).unsqueeze(0).unsqueeze(0).cuda())[0, 0].cpu().numpy() # m

    # Calculate ground and use for depth
    if use_float_gd:
        depth_new          = depth_center
    else:
        depth_ground       = ground_depth(intrinsics, extrinsics, cam_height, im_h, im_w, downsample, trans)[0].cpu().numpy() # H x W
        bx = bx.astype(np.int32)
        by = by.astype(np.int32)
        depth_center_quant = depth_ground[by, bx]
        depth_new          = depth_center_quant

    boxes_new[:, 12] = depth_new
    # boxes_new[:, 12] = depth_old

    # Write the new labels
    labels_new = np.concatenate((categories, boxes_new), axis= 1)
    write_lines(output_label_file, array_to_space_separated_string(labels_new))

    # Append (cx, cy, bx, by, by_ideal, depth_center)
    extras    =  np.concatenate((cx[:, np.newaxis], cy[:, np.newaxis], bx[:, np.newaxis], by[:, np.newaxis], by_ideal[:, np.newaxis], depth_center[:, np.newaxis]), axis= 1) # m x 6
    image_data= np.concatenate((image_data, extras), axis= 1)
    all_data  = image_data if all_data is None else np.concatenate((all_data, image_data), axis= 0)

# Save data
print("Saving to {}".format(npy_file_path))
save_numpy(npy_file_path, all_data)

print("\nRunning evaluation for 0.7 and 0.5 ....")
print("Output folder = {}".format(output_folder))
conf = {}
conf['dataset'] = {}
conf['dataset']['writelist'] = ['Car']

get_MAE(results_folder = output_folder, gt_folder= gt_folder, conf= None, use_logging= False, logger= None)
evaluate_kitti_results_verbose(gt_folder=gt_folder, test_dataset_name= "val", \
                                           results_folder=output_folder, test_iter= 140, conf= conf,\
                                           use_logging=False, logger=None)