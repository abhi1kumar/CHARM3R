"""
    Sample Run:
    python .py
"""
import os, sys
sys.path.append(os.getcwd())

import numpy as np
import glob
np.set_printoptions   (precision= 2, suppress= True)

from lib.helpers.rpn_util import filter_boxes_by_cars
from lib.helpers.file_io  import read_lines, write_lines, read_csv

input_folder   = "data/coda/training/label_2/"

def foo(split_txt_path, output_path):
    list_of_index  = read_lines(split_txt_path, strip= True)
    list_of_labels = []
    for i in list_of_index:
        list_of_labels.append(os.path.join(input_folder, i + ".txt"))

    lines_with_return_character = []

    relevant_cnt = 0
    for i, label_path in enumerate(list_of_labels):
        boxes = read_csv(label_path, use_pandas= True)
        img_id = os.path.basename(label_path).replace(".txt", "")
        if boxes is not None:
            mask1 = np.logical_or(boxes[:, 0] == "Car"       , boxes[:, 0] == "car")
            mask  = mask1
            if np.any(mask):
                relevant_cnt += 1
                lines_with_return_character += [img_id + "\n"]

        if (i+1) % 5000 == 0 or (i+1) == len(list_of_labels):
            print("{:5d}/{:5d} images done".format(relevant_cnt, i+1))

    write_lines(output_path, lines_with_return_character)

split_txt_path = "data/coda/ImageSets/train.txt"
output_path    = "data/coda/ImageSets/train_car.txt"
foo(split_txt_path, output_path)

split_txt_path = "data/coda/ImageSets/val.txt"
output_path    = "data/coda/ImageSets/val_car.txt"
foo(split_txt_path, output_path)