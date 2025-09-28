#
# Sample Run:
# bash tools/move_folders.sh run_030
#

set -exu

exp_name=$1

# ==================================================================================================
# Get folder paths
# ==================================================================================================
prefix="output/"
postfix1="/result_140"
in_folder=$prefix$exp_name$postfix1

height0_in="/pitch0"
height30_in="/height30"
height0_folder_in=$in_folder$height0_in
height30_folder_in=$in_folder$height30_in


postfix2="/result_carla"
height0_out="/height0"
out_folder=$prefix$exp_name$postfix2

height0_folder_out=$out_folder$height0_out
height30_folder_out=$out_folder$height30_in

# ==================================================================================================
# move folders
# ==================================================================================================
mv $height0_folder_in  $height0_folder_out
mv $height30_folder_in $height30_folder_out