"""
    Sample Run:
    python tools/merge_regression_ground_depth.py --folder output/gup_carlan_10 --folder2 output/run_1162 --iou 0.7 --ground 0.18
"""
import copy
import os, sys
sys.path.append(os.getcwd())

import os.path as osp
import glob
import numpy as np
import torch
import torch.nn as nn
import shutil

np.set_printoptions   (precision= 2, suppress= True)
torch.set_printoptions(precision= 2, sci_mode= False)

from lib.helpers.file_io import read_lines, read_csv, write_lines
from lib.helpers.rpn_util import iou
from lib.helpers.math_3d import project_3d_points_in_4D_format, backproject_2d_pixels_in_4D_format
from lib.helpers.rpn_util import get_MAE, evaluate_kitti_results_verbose
import argparse
from datetime import datetime
import logging
from lib.datasets.kitti_utils import get_calib_from_file

def create_logger(log_file):
    # Remove all handlers associated with the root logger object.
    # See https://stackoverflow.com/a/49202811
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(console)
    return logging.getLogger(__name__)

def boxes_to_uvZ(calib, boxes1):
    #       0  1   2      3   4   5   6    7    8    9   10   11   12   13    14
    # cls, -1, -1, alpha, x1, y1, x2, y2, h3d, w3d, l3d, x3d, y3d, z3d, ry3d, score
    boxes = copy.deepcopy(boxes1)
    h3d = boxes[:, 7]
    XYZ = boxes[:, 10:13]
    XYZ[:, 1] -= h3d/2.0  # N x 3
    uvZ = project_3d_points_in_4D_format(p2= calib, points_4d= XYZ.T, pad_ones= True).T [:, :3] # N x 3

    return uvZ

def uvZ_to_boxes(calib, uvZ, boxes1):
        #       0  1   2      3   4   5   6    7    8    9   10   11   12   13    14
    # cls, -1, -1, alpha, x1, y1, x2, y2, h3d, w3d, l3d, x3d, y3d, z3d, ry3d, score
    boxes = copy.deepcopy(boxes1)
    h3d = boxes[:, 7]
    XYZ = backproject_2d_pixels_in_4D_format(np.linalg.inv(calib), uvZ.T, pad_ones= True).T [:, :3]  # N x 3
    XYZ[:, 1] += h3d/2.0  # N x 3

    boxes[:, 10:13] = XYZ

    return boxes

def combine_two_folders(folder1, folder2, calib_f, out_data, logger):
    preds1 = sorted(glob.glob(folder1 + "/*.txt"))
    preds2 = sorted(glob.glob(folder2 + "/*.txt"))
    calibs = sorted(glob.glob(calib_f + "/*.txt"))

    assert len(preds1) == len(preds2)
    num_images = len(preds1)

    for i, (p1, p2, c) in enumerate(zip(preds1, preds2, calibs)):
        lines_with_return = ""
        lines1 = read_csv(p1, use_pandas= True)

        if lines1 is not None:
            class1 = lines1[:, 0:1]
            boxes1 = np.array(lines1[:, 1:])
            B = boxes1.shape[0]

            lines2 = read_csv(p2, use_pandas= True)
            if lines2 is not None:
                calib_3  = get_calib_from_file(c)['P2']
                calib = np.eye(4)
                calib[:3] = calib_3

                class2 = lines2[:, 0:1]
                boxes2 = np.array(lines2[:, 1:])

                uvZ1  = boxes_to_uvZ(calib, boxes1)
                uvZ2  = boxes_to_uvZ(calib, boxes2)

                # Compute IoU between boxes
                #       0   1   2      3   4   5   6    7    8    9   10   11   12   13    14
                # cls, -1, -1, alpha, x1, y1, x2, y2, h3d, w3d, l3d, x3d, y3d, z3d, ry3d, score
                overlap= iou(box_a= boxes1[:, 3:7], box_b= boxes2[:, 3:7])
                max_i = np.argmax(overlap, axis= 1)
                max_o = np.max   (overlap, axis= 1)
                flag  = max_o > iou_th

                wts   = np.zeros((B, 2))
                for j in range(B):
                    if flag[j]:
                        wts[j, 0] = wts[j, 1] = 0.5
                    else:
                        wts[j, 0] = 1.

                uvZ   = copy.deepcopy(uvZ1)
                uvZ[:, 2]   = wts[:, 0] * uvZ1[:, 2] + wts[:, 1] * (uvZ2[:, 2][max_i] + ground_bias)
                boxes_new = uvZ_to_boxes(calib, uvZ, boxes1)
            else:
                boxes_new = boxes1
        else:
            B = 0


        # Write to results folder
        out_f = osp.join(out_data, osp.basename(p1))
        os.makedirs(out_data, exist_ok= True)
        for j in range(B):
            lines_with_return += '{} 0.0 0'.format(class1[j][0])
            for k in range(2, 15):
                lines_with_return += ' {:.2f}'.format(boxes_new[j, k])
            # if j != B-1:
            lines_with_return += '\n'
        write_lines(out_f, lines_with_return)

        if (i+1) % 500 == 0 or (i+1) == num_images:
            logger.info("{} images done".format(i+1))

        # if (i+1) > 2000:
        #     break


    # Run evaluation
    gt_folder = calib_f.replace("calib", "label")
    results_folder = out_data

    # get MAE stats
    get_MAE(results_folder = results_folder, gt_folder= gt_folder, conf= None, use_logging= True, logger= logger)
    # get AP stats
    evaluate_kitti_results_verbose(gt_folder=gt_folder, test_dataset_name="val", \
                                       results_folder=results_folder, conf= cfg, \
                                       use_logging=True, logger= logger)


#================================================================
# Main starts here
#================================================================
parser = argparse.ArgumentParser(description='merging predictions')
parser.add_argument('--folder' , type=str, default = "output/gup_carlan_10", help='evaluate model on validation set')
parser.add_argument('--folder2', type=str, default=  "output/run_1162")
parser.add_argument('--iou', type=float, default= "0.7", help= 'matching threshold')
parser.add_argument('--ground', type=float, default= "0.0", help= 'ground bias to add to ground model')

args = parser.parse_args()

iou_th      = args.iou
ground_bias = args.ground
folder1  = args.folder
folder2 = args.folder2
rel_folder1  = osp.basename(args.folder)
rel_folder2 = osp.basename(args.folder2)
exp_parent_dir = osp.join("output", rel_folder1 + "_" + rel_folder2 + "_iou_" + "{:.1f}".format(iou_th) + "_ground_" + "{:.2f}".format(ground_bias) )
print("=> Making {}".format(exp_parent_dir))
os.makedirs(exp_parent_dir, exist_ok= True)

cfg= {'trainer': {}, 'dataset': {}}
cfg['dataset']['writelist'] = ['Car','Pedestrian','Cyclist']
cfg['trainer']['log_dir'] = exp_parent_dir
logger_dir     = os.path.join(exp_parent_dir, "log")
os.makedirs(exp_parent_dir, exist_ok=True)
os.makedirs(logger_dir, exist_ok=True)

list_of_heights = ["height-27", "height-24", "height-18", "height-12", "height-6", "height0", "height6", "height12", "height18", "height24", "height30"]

for height in list_of_heights:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = create_logger(os.path.join(logger_dir, timestamp))

    print("\n===================== Height= {} ========================".format(height))
    logger.info("IoU Threshold= {:.2f}".format(iou_th))
    logger.info("Ground Bias  = {:.2f}".format(ground_bias))
    data1   = osp.join(folder1, "result_carla", height, "data")
    data2   = osp.join(folder2, "result_carla", height, "data")
    calib_f = osp.join("data/carla/", height, "validation/calib")
    out_data = osp.join(exp_parent_dir, "result_carla", height, "data")
    if osp.exists(out_data):
        print("Removing {}".format(out_data))
        shutil.rmtree(out_data)
    os.makedirs(out_data, exist_ok= True)

    combine_two_folders(data1, data2, calib_f, out_data, logger)

# Python
command = "python tools/parse_log.py --folder=" + exp_parent_dir
os.system(command)