#===============================================================================
# Training Scripts
#===============================================================================

# ==== CARLA Val Split ====
# With GUP Net on DLA-34 backbone
CUDA_VISIBLE_DEVICES=0 python -u tools/train_val.py --config=experiments/carla_gup_dla34_height0.yaml
CUDA_VISIBLE_DEVICES=0 python -u tools/train_val.py --config=experiments/carla_charm3r_gup_dla34.yaml

# With DEVIANT on DLA-34 backbone
CUDA_VISIBLE_DEVICES=0 python -u tools/train_val.py --config=experiments/carla_dev_dla34_height0.yaml
CUDA_VISIBLE_DEVICES=0 python -u tools/train_val.py --config=experiments/carla_charm3r_gup_dla34.yaml

#===============================================================================
# Ablation Studies
#===============================================================================
# With GUP Net on ResNet-18 backbone
CUDA_VISIBLE_DEVICES=0 python -u tools/train_val.py --config=experiments/carla_gup_resnet18_height0.yaml
CUDA_VISIBLE_DEVICES=0 python -u tools/train_val.py --config=experiments/carla_charm3r_gup_resnet18.yaml

# With DEVIANT on ResNet-18 backbone
CUDA_VISIBLE_DEVICES=0 python -u tools/train_val.py --config=experiments/carla_dev_resnet18_height0.yaml
CUDA_VISIBLE_DEVICES=0 python -u tools/train_val.py --config=experiments/carla_charm3r_dev_resnet18.yaml
