#===============================================================================
# Inference Scripts
#===============================================================================

# ==== CARLA Val Split ====
# With GUP Net on DLA-34 backbone
bash tools/eval_on_all_heights.sh experiments/carla_gup_dla34_height0.yaml output/carla_gup_dla34_height0/checkpoints/checkpoint_epoch_140.pth 0
bash tools/eval_on_all_heights.sh experiments/carla_charm3r_gup_dla34.yaml output/carla_charm3r_gup_dla34/checkpoints/checkpoint_epoch_140.pth 0

# With DEVIANT on DLA-34 backbone
bash tools/eval_on_all_heights.sh experiments/carla_dev_dla34_height0.yaml output/carla_dev_dla34_height0/checkpoints/checkpoint_epoch_140.pth 0
bash tools/eval_on_all_heights.sh experiments/carla_charm3r_dev_dla34.yaml output/carla_charm3r_dev_dla34/checkpoints/checkpoint_epoch_140.pth 0
