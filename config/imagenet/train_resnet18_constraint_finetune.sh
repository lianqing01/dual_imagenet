#!/usr/bin/env bash

python -m torch.distributed.launch --nproc_per_node=$1 --master_port=$2 main_amp_constraint_finetune.py -a resnet_constraint18 --b 384 --workers 4  \
    ./data/imagenet  \
    --norm_layer $3 \
    --mixed_precision False \
    --lr $4 \
    --log_dir imagenet/constraint_20+norm_layer_+$3+lr$4+noise_$5+weight_+$6+warmup$7 \
    --epochs 50 \
    --constraint_lr 0.001 \
    --constraint_decay 1 \
    --lambda_constraint_weight $6 \
    --lambda_weight_mean 1 \
    --decrease_affine_lr 1 \
    --sample_noise True \
    --noise_data_dependent False \
    --noise_mean_std $5 \
    --noise_var_std $5 \
    --warmup_noise $7 \
    --resume $8

