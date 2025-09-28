#
# Sample Run:
# bash tools/eval_on_all_heights.sh experiments/run_041.yaml CHECKPOINT GPU
# bash tools/eval_on_all_heights.sh experiments/run_041.yaml output/run_041/checkpoints/checkpoint_epoch_140.pth 1
#

config=$1
checkpoint=$2

#exp_name=$(basename $(dirname $(dirname $checkpoint)))
temp=$(basename $config)
replace=""
exp_name=${temp//.yaml/$replace}

prefix="output/"
suffix="/result_*"
full_exp_name=$prefix$exp_name
full_name=$prefix$exp_name$suffix
echo $full_name
rm -rf $full_name

# Run inference over all height changes
CUDA_VISIBLE_DEVICES=$3 python -u tools/train_val.py --config=$1 --resume=$2 --ext height-27 -e
CUDA_VISIBLE_DEVICES=$3 python -u tools/train_val.py --config=$1 --resume=$2 --ext height-24 -e
CUDA_VISIBLE_DEVICES=$3 python -u tools/train_val.py --config=$1 --resume=$2 --ext height-18 -e
CUDA_VISIBLE_DEVICES=$3 python -u tools/train_val.py --config=$1 --resume=$2 --ext height-12 -e
CUDA_VISIBLE_DEVICES=$3 python -u tools/train_val.py --config=$1 --resume=$2 --ext height-6  -e

CUDA_VISIBLE_DEVICES=$3 python -u tools/train_val.py --config=$1 --resume=$2 --ext height0  -e
CUDA_VISIBLE_DEVICES=$3 python -u tools/train_val.py --config=$1 --resume=$2 --ext height6  -e
CUDA_VISIBLE_DEVICES=$3 python -u tools/train_val.py --config=$1 --resume=$2 --ext height12 -e
CUDA_VISIBLE_DEVICES=$3 python -u tools/train_val.py --config=$1 --resume=$2 --ext height18 -e
CUDA_VISIBLE_DEVICES=$3 python -u tools/train_val.py --config=$1 --resume=$2 --ext height24 -e
CUDA_VISIBLE_DEVICES=$3 python -u tools/train_val.py --config=$1 --resume=$2 --ext height30 -e

# Parse evaluation logs
python tools/parse_log.py --folder $full_exp_name
