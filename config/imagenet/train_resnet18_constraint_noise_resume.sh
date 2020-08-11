#!/usr/bin/env bash

python -m torch.distributed.launch --nproc_per_node=$1 --master_port=$2 main_amp_constraint_resume.py -a resnet_constraint18 --b 512 --workers 4  \
    ./data/imagenet  \
    --norm_layer $3 \
    --mixed_precision False \
    --lr $6 \
    --log_dir imagenet/constraint_20+norm_layer_+$3+noise_$4+warmup_+$5+lr_$6 \
    --epochs 120 \
    --constraint_lr 0.001 \
    --constraint_decay 1 \
    --lambda_constraint_weight 0.5 \
    --lambda_weight_mean 1 \
    --decrease_affine_lr 1 \
    --sample_noise True \
    --noise_data_dependent False \
    --noise_mean_std $4 \
    --noise_var_std $4 \
    --warmup_noise $5 \
    --resume $7

