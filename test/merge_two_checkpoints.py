"""
    Sample Run:
    python .py
"""
import copy
import os, sys
sys.path.append(os.getcwd())

import numpy as np
import torch
np.set_printoptions   (precision= 2, suppress= True)

def replace_layer_params(ckpt_bline, ckpt_oracle, replace_keys_list, model_output):
    weights_bline  = ckpt_bline['model_state']
    weights_oracle = ckpt_oracle['model_state']

    weights_output = copy.deepcopy(weights_bline)
    keys_list = list(weights_bline.keys())

    for k in keys_list:
        if any(replace_key in k for replace_key in replace_keys_list):
            print("Copying oracle {}".format(k))
            weights_output[k] = weights_oracle[k]

    ckpt_output = copy.deepcopy(ckpt_bline)
    ckpt_output['model_state'] = weights_output

    print("==> Saving to checkpoint '{}'".format(model_output))
    torch.save(ckpt_output, model_output)


model_bline  = "output/gup_carla/checkpoints/checkpoint_epoch_140.pth"
model_no_gap = "output/gup_carla_height0_6_25k/checkpoints/checkpoint_epoch_140.pth"

device       = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ckpt_bline   = torch.load(model_bline, map_location= device)
ckpt_oracle  = torch.load(model_no_gap, map_location= device)

# model_output = "output/gup_carla/checkpoints/checkpoint_oracle_head.pth"
# replace_keys_list = ['heatmap', 'offset_2d', 'size_2d', 'size_3d', 'offset_3d', 'depth', 'heading']
# replace_layer_params(ckpt_bline, ckpt_oracle, replace_keys_list, model_output)

# replace_keys_list = ['feat_up', 'heatmap', 'offset_2d', 'size_2d', 'size_3d', 'offset_3d', 'depth', 'heading']
# model_output = "output/gup_carla/checkpoints/checkpoint_oracle_head_fpn.pth"
# replace_layer_params(ckpt_bline, ckpt_oracle, replace_keys_list, model_output)

# replace_keys_list = ['backbone', 'feat_up', 'heatmap', 'offset_2d', 'size_2d', 'size_3d', 'offset_3d', 'depth', 'heading']
# model_output = "output/gup_carla/checkpoints/checkpoint_oracle.pth"
# replace_layer_params(ckpt_bline, ckpt_oracle, replace_keys_list, model_output)

replace_keys_list = ['backbone']
model_output      = "output/gup_carla/checkpoints/checkpoint_oracle_backbone.pth"
replace_layer_params(ckpt_bline, ckpt_oracle, replace_keys_list, model_output)