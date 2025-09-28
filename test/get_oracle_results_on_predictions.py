"""
    Sample Run:
    python test/get_oracle_results_on_predictions.py
"""
import os, sys
sys.path.append(os.getcwd())

import numpy as np

from lib.helpers.oracle_helper import run_oracle_on_predictions
np.set_printoptions   (precision= 4, suppress= True)

list_of_heights  = ["height-27", "height0", "height30"]
list_of_cat      = ['Car']
update_cat       = ['Car']
dist_max_th_list = {'Car': 4}
prediction_folder= "output/gup_carlan_10/result_carla/"

for height in list_of_heights:
    gt_folder     = os.path.join("data/carla/", height + "/validation/label")
    result_folder = os.path.join(prediction_folder, height + "/data/")

    oracle_input_list  = ['x3d']
    run_oracle_on_predictions(gt_folder, result_folder, oracle_input_list= oracle_input_list, list_of_cat= list_of_cat, update_cat= update_cat, dist_max_th_list= dist_max_th_list)

    oracle_input_list  = ['y3d']
    run_oracle_on_predictions(gt_folder, result_folder, oracle_input_list= oracle_input_list, list_of_cat= list_of_cat, update_cat= update_cat, dist_max_th_list= dist_max_th_list)

    oracle_input_list  = ['z3d']
    run_oracle_on_predictions(gt_folder, result_folder, oracle_input_list= oracle_input_list, list_of_cat= list_of_cat, update_cat= update_cat, dist_max_th_list= dist_max_th_list)

    oracle_input_list  = ['x3d', 'y3d', 'z3d']
    run_oracle_on_predictions(gt_folder, result_folder, oracle_input_list= oracle_input_list, list_of_cat= list_of_cat, update_cat= update_cat, dist_max_th_list= dist_max_th_list)

    oracle_input_list  = ['l3d', 'w3d', 'h3d']
    run_oracle_on_predictions(gt_folder, result_folder, oracle_input_list= oracle_input_list, list_of_cat= list_of_cat, update_cat= update_cat, dist_max_th_list= dist_max_th_list)

    oracle_input_list  = ['x3d', 'z3d', 'y3d', 'l3d', 'w3d', 'h3d']
    run_oracle_on_predictions(gt_folder, result_folder, oracle_input_list= oracle_input_list, list_of_cat= list_of_cat, update_cat= update_cat, dist_max_th_list= dist_max_th_list)

    oracle_input_list  = ['x3d', 'z3d', 'y3d', 'l3d', 'w3d', 'h3d', 'ry3d']
    run_oracle_on_predictions(gt_folder, result_folder, oracle_input_list= oracle_input_list, list_of_cat= list_of_cat, update_cat= update_cat, dist_max_th_list= dist_max_th_list)