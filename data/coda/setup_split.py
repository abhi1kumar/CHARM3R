import numpy as np
import sys
import os
import shutil
import re

# stop python from writing so much bytecode
sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

def mkdir_if_missing(directory, delete_if_exist=False):
    """
    Recursively make a directory structure even if missing.

    if delete_if_exist=True then we will delete it first
    which can be useful when better control over initialization is needed.
    """

    if delete_if_exist and os.path.exists(directory): shutil.rmtree(directory)

    # check if not exist, then make
    if not os.path.exists(directory):
        os.makedirs(directory)

def make_symlink_or_copy(src_path, intended_path, MAKE_SYMLINK = True):
    if not os.path.exists(intended_path):
        if MAKE_SYMLINK:
            os.symlink(src_path, intended_path)
        else:
            command = "cp " + src_path + " " + intended_path
            os.system(command)

def link_original_data(org_file_path, new_file_path, org_split_folders, new_split_folders):
    # mkdirs
    mkdir_if_missing(new_split_folders['cal'])
    mkdir_if_missing(new_split_folders['ims'])
    mkdir_if_missing(new_split_folders['lab'])
    mkdir_if_missing(new_split_folders['dep'])
    #mkdir_if_missing(new_split_folders['sem'])
    mkdir_if_missing(new_split_folders['dsine'])
    #mkdir_if_missing(new_split_folders['metnor'])

    print("=> Reading {}".format(org_file_path))
    print("=> Writing {}".format(new_file_path))
    text_file_new = open(new_file_path, 'w')

    text_file     = open(org_file_path, 'r')
    text_lines = text_file.readlines()
    text_file.close()

    for imind, line in enumerate(text_lines):
        parsed = line.strip()

        if parsed is not None:
            id = parsed
            if "val" in org_file_path:
                cnt  = imind + 19511
            else:
                cnt  = imind
            new_id = '{:06d}'.format(cnt)

            org_calib_path = os.path.join(org_split_folders['cal'], id + '.txt')
            org_label_path = os.path.join(org_split_folders['lab'], id + '.txt')
            org_image_path = os.path.join(org_split_folders['ims'], id + '.jpg')
            org_depth_path = os.path.join(org_split_folders['dep'], id + '.bin')
            #org_seman_path = os.path.join(org_split_folders['sem'].replace("seman", ""), f, "seman", id + '.npy')
            org_dsine_path = os.path.join(org_split_folders['dsine'], id + '.png')
            #org_metnor_path = os.path.join(org_split_folders['metnor'].replace("metnor", ""), f, "metnor", id + '.png')

            # If any of the calib/label/image is missing
            # if not os.path.exists(org_calib_path) or not os.path.exists(org_label_path) or not os.path.exists(org_image_path):
            #     print("{} not found ...".format(parsed))
            #     imind += 1
            #     continue

            new_calib_path = os.path.join(new_split_folders['cal'], str(new_id) + '.txt')
            new_label_path = os.path.join(new_split_folders['lab'], str(new_id) + '.txt')
            new_image_path = os.path.join(new_split_folders['ims'], str(new_id) + '.jpg')
            new_depth_path = os.path.join(new_split_folders['dep'], str(new_id) + '.bin')
            #new_seman_path = os.path.join(new_split_folders['sem'], str(new_id) + '.npy')
            new_dsine_path = os.path.join(new_split_folders['dsine'], str(new_id) + '.png')
            #new_metnor_path= os.path.join(new_split_folders['metnor'], str(new_id) + '.png')

            make_symlink_or_copy(org_calib_path, new_calib_path)
            make_symlink_or_copy(org_label_path, new_label_path)
            make_symlink_or_copy(org_image_path, new_image_path)
            make_symlink_or_copy(org_depth_path, new_depth_path)
            #make_symlink_or_copy(org_seman_path, new_seman_path)
            make_symlink_or_copy(org_dsine_path, new_dsine_path)
            #make_symlink_or_copy(org_metnor_path, new_metnor_path)

            text_file_new.write(new_id + '\n')
            imind += 1

        if imind % 500 == 0 or line == text_lines[-1]:
            print("{} images done...".format(imind))


    text_file_new.close()

#===================================================================================================
# Main starts here
#===================================================================================================
curr_folder = os.getcwd()
org_train_file  = os.path.join(curr_folder, 'ImageSets/train_org.txt')
org_val_file    = os.path.join(curr_folder, 'ImageSets/val_org.txt')
org_test_file   = os.path.join(curr_folder, 'ImageSets/test_org.txt')
new_train_file  = os.path.join(curr_folder, 'ImageSets/train.txt')
new_val_file    = os.path.join(curr_folder, 'ImageSets/val.txt')
new_test_file   = os.path.join(curr_folder, 'ImageSets/test.txt')

print("")
org_train_folders = dict()
org_train_folders['cal'] = os.path.join(curr_folder, 'training_org', 'calib')
org_train_folders['ims'] = os.path.join(curr_folder, 'training_org', 'image_2')
org_train_folders['lab'] = os.path.join(curr_folder, 'training_org', 'label_2')
org_train_folders['dep'] = os.path.join(curr_folder, 'training_org', 'velodyne')
#org_train_folders['sem'] = os.path.join(curr_folder, 'training_org', 'seman')
org_train_folders['dsine'] = os.path.join(curr_folder, 'training_org', 'dsine')
#org_train_folders['metnor'] = os.path.join(curr_folder, 'training_org', 'metnor')

new_train_folders = dict()
split = "training"
new_train_folders['cal'] = os.path.join(curr_folder, split, 'calib')
new_train_folders['ims'] = os.path.join(curr_folder, split, 'image_2')
new_train_folders['lab'] = os.path.join(curr_folder, split, 'label_2')
new_train_folders['dep'] = os.path.join(curr_folder, split, 'velodyne')
#new_train_folders['sem'] = os.path.join(curr_folder, split, 'seman')
new_train_folders['dsine'] = os.path.join(curr_folder, split, 'dsine')
#new_train_folders['metnor'] = os.path.join(curr_folder, split, 'metnor')

org_test_folders = dict()
org_test_folders['cal'] = os.path.join(curr_folder, 'testing_org', 'calib')
org_test_folders['ims'] = os.path.join(curr_folder, 'testing_org', 'image_2')
org_test_folders['lab'] = os.path.join(curr_folder, 'testing_org', 'label_2')
org_test_folders['dep'] = os.path.join(curr_folder, 'testing_org', 'velodyne')
#org_test_folders['sem'] = os.path.join(curr_folder, 'testing_org', 'seman')
org_test_folders['dsine'] = os.path.join(curr_folder, 'testing_org', 'dsine')
#org_test_folders['metnor'] = os.path.join(curr_folder, 'testing_org', 'metnor')


new_test_folders = dict()
split = "testing"
new_test_folders['cal'] = os.path.join(curr_folder, split, 'calib')
new_test_folders['ims'] = os.path.join(curr_folder, split, 'image_2')
new_test_folders['lab'] = os.path.join(curr_folder, split, 'label_2')
new_test_folders['dep'] = os.path.join(curr_folder, split, 'velodyne')
#new_test_folders['sem'] = os.path.join(curr_folder, split, 'seman')
new_test_folders['dsine'] = os.path.join(curr_folder, split, 'dsine')
#new_test_folders['metnor'] = os.path.join(curr_folder, split, 'metnor')

#===================================================================================================
# Link train
#===================================================================================================
print('=============== Linking CODa train =======================')
link_original_data(org_file_path= org_train_file, new_file_path= new_train_file, org_split_folders = org_train_folders, new_split_folders= new_train_folders)

#===================================================================================================
# Link val
#===================================================================================================
print('=============== Linking CODa val =======================')
link_original_data(org_file_path= org_val_file, new_file_path= new_val_file, org_split_folders = org_train_folders, new_split_folders= new_train_folders)

#===================================================================================================
# Link val
#===================================================================================================
print('=============== Linking CODa test =======================')
link_original_data(org_file_path= org_test_file, new_file_path= new_test_file, org_split_folders = org_test_folders, new_split_folders= new_test_folders)

print('Done')
