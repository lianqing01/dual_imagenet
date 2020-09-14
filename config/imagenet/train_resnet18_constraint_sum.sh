#!/usr/bin/env bash

python -m torch.distributed.launch --nproc_per_node=$1 --master_port=$2 main_amp_constraint_sum_draw.py -a resnet_constraint18 --b 256 --workers 4  \
    ./data/imagenet  \
    --norm_layer $3 \
    --mixed_precision False \
    --opt-level O1 \
    --log_dir imagenet/constraint_20+norm_layer_+$3+cweight+$4+cdecay+$5\
    --epochs 120 \
    --constraint_lr 0.1 \
    --constraint_decay $5 \
    --lambda_constraint_weight $4 \
    --lambda_weight_mean 1 \
    --decrease_affine_lr 1 \
    --sample_noise False \
    --noise_data_dependent False \
    --noise_mean_std 0 \
    --noise_var_std 0 \
    ${@:6}

