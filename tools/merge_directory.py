"""
    Sample Run:
    python tools/merge_directory.py

    Merges data from multiple directories to produce a new folder
"""
import os, sys
sys.path.append(os.getcwd())

import numpy as np
np.set_printoptions   (precision= 4, suppress= True)

import argparse

from lib.helpers.file_io import read_lines, write_lines

def make_symlink_or_copy(src_path, intended_path, MAKE_SYMLINK = True):
    if not os.path.exists(intended_path):
        if MAKE_SYMLINK:
            os.symlink(src_path, intended_path)
        else:
            command = "cp " + src_path + " " + intended_path
            os.system(command)


def merge_folders(list_of_extrinsics, new_name, pwd):
    print("=======================================================================================")
    print("Name = {}".format(new_name))
    print("=======================================================================================")
    for split in ["training", "validation"]:
        base_folder  = os.path.join(pwd, "data/carla/", new_name, split)
        calib_folder = os.path.join(base_folder, "calib")
        depth_folder = os.path.join(base_folder, "depth")
        dsine_folder = os.path.join(base_folder, "dsine")
        image_folder = os.path.join(base_folder, "image")
        label_folder = os.path.join(base_folder, "label")
        lidar_folder = os.path.join(base_folder, "lidar")
        metnor_folder= os.path.join(base_folder, "metnor")
        seman_folder = os.path.join(base_folder, "seman")

        # ==========================================================================================
        # Make folders
        os.makedirs(calib_folder, exist_ok= True)
        os.makedirs(depth_folder, exist_ok= True)
        os.makedirs(dsine_folder, exist_ok= True)
        os.makedirs(image_folder, exist_ok= True)
        os.makedirs(label_folder, exist_ok= True)
        os.makedirs(lidar_folder, exist_ok= True)
        os.makedirs(metnor_folder, exist_ok= True)
        os.makedirs(seman_folder, exist_ok= True)

        if split == "training":
            id_file = "data/carla/ImageSets/train.txt"
            out_id_file = "data/carla/ImageSets/train_" + new_name + ".txt"
        else:
            id_file = "data/carla/ImageSets/val.txt"
            out_id_file = "data/carla/ImageSets/val_" + new_name + ".txt"

        id_list   = read_lines(id_file)
        out_id_list = []
        split_cnt = 0
        for ext in list_of_extrinsics:
            curr_base_folder  = os.path.join(pwd, "data/carla/", ext, split)
            curr_calib_folder = os.path.join(curr_base_folder, "calib")
            curr_depth_folder = os.path.join(curr_base_folder, "depth")
            curr_dsine_folder = os.path.join(curr_base_folder, "dsine")
            curr_image_folder = os.path.join(curr_base_folder, "image")
            curr_label_folder = os.path.join(curr_base_folder, "label")
            curr_lidar_folder = os.path.join(curr_base_folder, "lidar")
            curr_metnor_folder= os.path.join(curr_base_folder, "metnor")
            curr_seman_folder = os.path.join(curr_base_folder, "seman")
            print("Folder = {}".format(curr_base_folder))

            # Link files
            for id in id_list:
                out_id = "{:06d}".format(split_cnt)

                src_path  = os.path.join(curr_calib_folder,     id + ".txt")
                int_path  = os.path.join(     calib_folder, out_id + ".txt")
                make_symlink_or_copy(src_path, int_path)

                src_path  = os.path.join(curr_depth_folder,     id + ".npy")
                int_path  = os.path.join(     depth_folder, out_id + ".npy")
                make_symlink_or_copy(src_path, int_path)

                src_path  = os.path.join(curr_dsine_folder,     id + ".png")
                int_path  = os.path.join(     dsine_folder, out_id + ".png")
                make_symlink_or_copy(src_path, int_path)

                src_path  = os.path.join(curr_image_folder,     id + ".jpg")
                int_path  = os.path.join(     image_folder, out_id + ".jpg")
                make_symlink_or_copy(src_path, int_path)

                src_path  = os.path.join(curr_label_folder,     id + ".txt")
                int_path  = os.path.join(     label_folder, out_id + ".txt")
                make_symlink_or_copy(src_path, int_path)

                src_path  = os.path.join(curr_lidar_folder,     id + ".npy")
                int_path  = os.path.join(     lidar_folder, out_id + ".npy")
                make_symlink_or_copy(src_path, int_path)

                src_path  = os.path.join(curr_metnor_folder,     id + ".png")
                int_path  = os.path.join(     metnor_folder, out_id + ".png")
                make_symlink_or_copy(src_path, int_path)

                src_path  = os.path.join(curr_seman_folder,     id + ".npy")
                int_path  = os.path.join(     seman_folder, out_id + ".npy")
                make_symlink_or_copy(src_path, int_path)

                out_id_list.append( out_id + "\n")
                split_cnt += 1


        write_lines(out_id_file, out_id_list)
        print("Done.")

# ==================================================================================================
#
# ====

pwd = os.getcwd()
merge_folders(list_of_extrinsics= ["pitch0"   , "height6"  ], new_name= "height0_6"  , pwd= pwd)
merge_folders(list_of_extrinsics= ["height6"  , "height12" ], new_name= "height6_12" , pwd= pwd)
merge_folders(list_of_extrinsics= ["pitch0"   , "height18" ], new_name= "height0,18" , pwd= pwd)
merge_folders(list_of_extrinsics= ["pitch0"   , "height30" ], new_name= "height0,30" , pwd= pwd)
merge_folders(list_of_extrinsics= ["pitch0"   , "height-27"], new_name= "height0,-27", pwd= pwd)

merge_folders(list_of_extrinsics= ["pitch0"   , "height6"  , "height12"], new_name= "height0_12" , pwd= pwd)
merge_folders(list_of_extrinsics= ["height-6" , "pitch0"   , "height6" ], new_name= "height-6_6" , pwd= pwd)
merge_folders(list_of_extrinsics= ["height-12", "height-6" , "pitch0"  ], new_name= "height-12_0", pwd= pwd)

merge_folders(list_of_extrinsics= ["pitch0"   , "height6"  , "height12", "height18"], new_name= "height0_18", pwd= pwd)
merge_folders(list_of_extrinsics= ["height-6" , "pitch0"   , "height6" , "height12"], new_name= "height-6_12", pwd= pwd)
merge_folders(list_of_extrinsics= ["height-18", "height-12", "height-6", "pitch0"  ], new_name= "height-18_0", pwd= pwd)
merge_folders(list_of_extrinsics= ["pitch0"   , "height6"  , "height12", "height18", "height24"], new_name= "height0_24", pwd= pwd)
merge_folders(list_of_extrinsics= ["pitch0"   , "height6"  , "height12", "height18", "height24", "height30"], new_name= "height0_30", pwd= pwd)