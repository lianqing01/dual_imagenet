#!/usr/bin/env bash

python -m torch.distributed.launch --nproc_per_node=$1 --master_port=$2 main_amp_constraint.py -a $3 --b 512 --workers 4  \
    ./data/imagenet  \
    --norm_layer $4 \
    --mixed_precision False \
    --opt-level O1 \
    --log_dir imagenet/$3+norm_layer_+$4+cweight+$5+cdecay+$6\
    --epochs 120 \
    --constraint_lr 0.1 \
    --constraint_decay $6 \
    --lambda_constraint_weight $5 \
    --lambda_weight_mean 1 \
    --decrease_affine_lr 1 \
    --sample_noise False \
    --noise_data_dependent False \
    --noise_mean_std 0 \
    --noise_var_std 0 \
    ${@:7}

