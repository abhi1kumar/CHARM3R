#
# Sample Run:
# bash tools/transfer_hpcc_eval.sh run_030
#

set -xu

exp_name=$1

# ==================================================================================================
# Make desktop folder
# ==================================================================================================
prefix="output/"
postfix="/result_carla"
desktop_folder=$prefix$exp_name$postfix

mkdir -p $desktop_folder

# ==================================================================================================
# Copy yaml
# ==================================================================================================
remote_path="/mnt/home/kumarab6/project/DEVIANT_CARLA/"
exp_prefix="experiments/"
exp_postfix=".yaml"
exp_yaml=$exp_prefix$exp_name$exp_postfix

remote_exp_yaml=$remote_path$exp_yaml
rsync -qazu kumarab6@rsync.hpcc.msu.edu:$remote_exp_yaml experiments/

# ==================================================================================================
# Copy checkpoint
# ==================================================================================================
ckp_postfix="/checkpoints/"
ckp_epoch="checkpoint_epoch_140.pth"
ckp_folder=$prefix$exp_name$ckp_postfix
ckp_file=$prefix$exp_name$ckp_postfix$ckp_epoch

mkdir -p $ckp_folder

remote_ckp_file=$remote_path$ckp_file
rsync -qazu kumarab6@rsync.hpcc.msu.edu:$remote_ckp_file $ckp_folder

# ==================================================================================================
# Run evaluation
# ==================================================================================================
bash tools/eval_on_all_heights.sh $exp_yaml $ckp_file 0
