#
# Sample Run:
# bash tools/transfer_cvl_eval_height0.sh run_030
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
remote_path="/user/kumarab6/cvl/project/DEVIANT_CARLA/"
exp_prefix="experiments/"
exp_postfix=".yaml"
exp_yaml=$exp_prefix$exp_name$exp_postfix

remote_exp_yaml=$remote_path$exp_yaml
rsync -qazu kumarab6@cvl11.cse.msu.edu:$remote_exp_yaml experiments/

# ==================================================================================================
# Copy checkpoint
# ==================================================================================================
ckp_postfix="/checkpoints/"
ckp_epoch="checkpoint_epoch_140.pth"
ckp_folder=$prefix$exp_name$ckp_postfix
ckp_file=$prefix$exp_name$ckp_postfix$ckp_epoch

mkdir -p $ckp_folder

remote_ckp_file=$remote_path$ckp_file
rsync -qazu kumarab6@cvl11.cse.msu.edu:$remote_ckp_file $ckp_folder

# ==================================================================================================
# Copy predictions
# ==================================================================================================
ext_config="/height0"
remote_folder=$remote_path$desktop_folder$ext_config
rsync -qazu kumarab6@cvl11.cse.msu.edu:$remote_folder $desktop_folder

ext_config="/height30"
remote_folder=$remote_path$desktop_folder$ext_config
rsync -qazu kumarab6@cvl11.cse.msu.edu:$remote_folder $desktop_folder

# ==================================================================================================
# Run evaluation
# ==================================================================================================
#python -u tools/train_val.py --resume=output/gup_carla_height6/checkpoints/checkpoint_epoch_140.pth --ext height0 --config=$exp_yaml -e
bash tools/eval_on_all_heights.sh $exp_yaml $ckp_file 0