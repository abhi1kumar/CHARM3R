"""
    Sample Run:
    python tools/parse_log.py --folder output/run_001

    Parse evaluation logs at different height and show output string
"""
import os, sys
sys.path.append(os.getcwd())
import numpy as np
import argparse
import glob

np.set_printoptions   (precision= 2, suppress= True)
from datetime import datetime
from lib.helpers.file_io import read_lines, write_lines

NUM_VAR = 3

def search_phrase(lines, phrase):
    line_num = []
    for i, l in enumerate(lines):
        if phrase in l:
            line_num.append(i)

    return line_num

def parse_log(filepath):
    lines    = read_lines(filepath)
    results = np.zeros((NUM_VAR, ))

    # Search for word
    line_num = search_phrase(lines, "test_iter None car 3d")
    for i, lindex in enumerate(line_num):
        words = lines[lindex].split(" ")
        word_ind = search_phrase(words, "mod:")[0]
        results[i] = float(words[word_ind+1].replace(",", ""))*100.0

    # Search for median MAE
    line_num = search_phrase(lines, " Median ")
    for i, lindex in enumerate(line_num):
        words = lines[lindex+3].split(" ")
        results[NUM_VAR-1] = float(words[8])

    return results


#================================================================
# Main starts here
#================================================================
parser = argparse.ArgumentParser(description='implementation of GUPNet')
parser.add_argument('--folder', type=str, default=  "output/gup_carla")
args = parser.parse_args()
log_folder = os.path.join(args.folder, "log")

num_files = 11
det_results = np.zeros((NUM_VAR+1, num_files)).astype(np.float64)
t = np.arange(-5, 6)* 0.15
t[0] = -0.67  # height-27
det_results[0] = t

list_of_files = sorted(glob.glob(log_folder + "/*"), reverse= True)
list_of_files = list_of_files[:num_files][::-1]
for i, filepath in enumerate(list_of_files):
    det_results[1:, i] = parse_log(filepath)
print(det_results)

det_results_2 = det_results.tolist()

det_results     = det_results[1:]
det_results     = np.round(det_results, 2).reshape(-1)
det_results_str = [str(x) for x in det_results]
output_str    = ",".join(det_results_str)
print(output_str)

# Write to a log file
lines_with_return_character  = '\n'.join('\t'.join('{:.2f}'.format(x) for x in y) for y in det_results_2)
lines_with_return_character += '\n'
lines_with_return_character += output_str + '\n'
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file  = os.path.join(log_folder, "eval_on_all_heights_" + timestamp)
write_lines(log_file, lines_with_return_character)