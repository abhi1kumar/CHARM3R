import os, sys

import numpy as np
import copy
import pandas as pd
import cv2
import glob
import shutil

from scipy.spatial import distance
from lib.helpers.file_io import imread, read_csv, read_lines, write_lines
from lib.helpers.rpn_util import evaluate_kitti_results_verbose, get_MAE

def run_oracle_on_predictions(gt_folder, result_folder, oracle_input_list= ['h3d', 'y3d'],
                              list_of_cat=['Car', 'Building'], update_cat= ['Car', 'Building'],
                              dist_max_th_list= {'Car': 1, 'Building': 7}):
    # Runs oracle on predictions
    print("\nList of categories     = {}".format(",".join(list_of_cat)))
    print("\nUpdate categories      = {}".format(",".join(update_cat)))
    print("Update Param by Oracle = {}".format(",".join(oracle_input_list)))
    print("GT Label folder   = {}".format(gt_folder))
    print("Pred Result folder= {}".format(result_folder))
    output_folder      = result_folder.replace("result_", "result_oracle_" + "_".join(oracle_input_list))
    if os.path.exists(output_folder):
        print("Remove existing folder = {}".format(output_folder))
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    conf = {}
    conf['dataset']= {'writelist': ['Car']}

    list_of_pred_files = sorted(glob.glob(os.path.join(result_folder, "*.txt")))
    total_boxes        = 0
    updated_boxes      = 0

    for i, pred_file_path in enumerate(list_of_pred_files):
        index     = os.path.basename(pred_file_path)#.replace(".txt", "")
        gt_file   = os.path.join(gt_folder, index)

        all_pred  = read_csv(pred_file_path, ignore_warnings= True, use_pandas= True)
        all_gt    = read_csv(gt_file, ignore_warnings= True, use_pandas= True)
        lines_out = []
        if all_pred is not None:
            for c, cat in enumerate(list_of_cat):
                all_pred_temp = all_pred[all_pred[:, 0] == cat]
                num_boxes     = all_pred_temp.shape[0]
                if num_boxes > 0:
                    cls_pred      = all_pred_temp[:, 0]
                    data_pred     = all_pred_temp[:, 1:].astype(np.float32)
                    #       0  1   2      3   4   5   6    7    8    9   10   11   12   13   14
                    # cls, -1, -1, alpha, x1, y1, x2, y2, h3d, w3d, l3d, x3d, y3d, z3d, ry3d score
                    alpha_pred = data_pred[:, 2]
                    x1_pred  = data_pred[:, 3]
                    y1_pred  = data_pred[:, 4]
                    x2_pred  = data_pred[:, 5]
                    y2_pred  = data_pred[:, 6]
                    h3d_pred = data_pred[:, 7]
                    w3d_pred = data_pred[:, 8]
                    l3d_pred = data_pred[:, 9]
                    x3d_pred = data_pred[:, 10]
                    y3d_pred = data_pred[:, 11]
                    z3d_pred = data_pred[:, 12]
                    ry3d_pred = data_pred[:,13]
                    scor_pred = data_pred[:, 14]

                    if cat in update_cat and all_gt is not None:
                        all_gt_temp = all_gt[all_gt[:, 0] == cat]
                        if all_gt_temp.shape[0] > 0:
                            cls_gt     = all_gt[:, 0]
                            data_gt    = all_gt[:, 1:].astype(np.float32)
                            #       0  1   2      3   4   5   6    7    8    9   10   11   12   13
                            # cls, -1, -1, alpha, x1, y1, x2, y2, h3d, w3d, l3d, x3d, y3d, z3d, ry3d
                            alpha = data_gt[:, 2]
                            x1  = data_gt[:, 3]
                            y1  = data_gt[:, 4]
                            x2  = data_gt[:, 5]
                            y2  = data_gt[:, 6]
                            h3d = data_gt[:, 7]
                            w3d = data_gt[:, 8]
                            l3d = data_gt[:, 9]
                            x3d = data_gt[:, 10]
                            y3d = data_gt[:, 11]
                            z3d = data_gt[:, 12]
                            ry3d = data_gt[:,13]

                            # Computer intersection between pred and ground truth boxes
                            center_bev_pred = data_pred[:, [10, 12]]
                            center_bev_gt   = data_gt[:, [10, 12]]

                            dist_mat          = distance.cdist(XA= center_bev_pred, XB= center_bev_gt, metric='cityblock')
                            dist_min          = np.min   (dist_mat, axis= 1)
                            dist_min_index    = np.argmin(dist_mat, axis= 1)
                            update_index_bool = dist_min < dist_max_th_list[cat]

                            update_index   = np.arange(num_boxes)[update_index_bool]
                            dist_min_index = dist_min_index      [update_index_bool]

                            updated_boxes += update_index.shape[0]

                            # Update the entries
                            if 'l3d' in oracle_input_list:
                                l3d_pred[update_index] = l3d[dist_min_index]
                            if 'w3d' in oracle_input_list:
                                w3d_pred[update_index] = w3d[dist_min_index]
                            if 'h3d' in oracle_input_list:
                                h3d_pred[update_index] = h3d[dist_min_index]
                            if 'x3d' in oracle_input_list:
                                x3d_pred[update_index] = x3d[dist_min_index]
                            if 'y3d' in oracle_input_list:
                                y3d_pred[update_index] = y3d[dist_min_index]
                            if 'z3d' in oracle_input_list:
                                z3d_pred[update_index] = z3d[dist_min_index]
                            if 'ry3d' in oracle_input_list:
                                ry3d_pred[update_index] = ry3d[dist_min_index]
                            # if 'score' in oracle_input_list:
                            #     scor_pred[update_index] = 1

                    for j, line in enumerate(np.arange(num_boxes)):
                        total_boxes += 1
                        cls_box   = cat
                        x1_box    = x1_pred[j]
                        y1_box    = y1_pred[j]
                        x2_box    = x2_pred[j]
                        y2_box    = y2_pred[j]
                        h3d_box   = h3d_pred[j]
                        y3d_box   = y3d_pred[j]

                        l3d_box   = l3d_pred[j]
                        w3d_box   = w3d_pred[j]
                        x3d_box   = x3d_pred[j]
                        z3d_box   = z3d_pred[j]
                        alpha_box = alpha_pred[j]
                        ry3d_box  = ry3d_pred[j]
                        score_box = scor_pred[j]

                        output_str = ("{} {:.2f} {:1d} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}\n"
                        .format(cls_box, -1, -1, alpha_box,
                                x1_box, y1_box, x2_box, y2_box,
                                h3d_box, w3d_box, l3d_box,
                                x3d_box, y3d_box, z3d_box, ry3d_box, score_box))
                        lines_out.append(output_str)
        write_lines(path= os.path.join(output_folder, os.path.basename(gt_file)), lines_with_return_character= lines_out)

        if (i+1)%500 == 0 or i == len(list_of_pred_files)-1:
            print("{:5d} images done {:5d} tot_bx {:5d} update_bx.".format(i+1, total_boxes, updated_boxes))

    get_MAE(results_folder = output_folder, gt_folder= gt_folder, conf= None, use_logging= False, logger= None)
    evaluate_kitti_results_verbose(gt_folder= gt_folder, test_dataset_name="val1",
                                   results_folder= output_folder.replace("/data",""), conf= conf,
                                   use_logging= False, logger= None)