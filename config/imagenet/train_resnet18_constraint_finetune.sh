#!/usr/bin/env bash

python -m torch.distributed.launch --nproc_per_node=$1 --master_port=$2 main_amp_constraint_finetune.py -a resnet_constraint18 --b 256 --workers 4  \
    ./data/imagenet  \
    --norm_layer $3 \
    --mixed_precision False \
    --opt-level O1 \
    --lr $4 \
    --log_dir imagenet/constraint_20+norm_layer_+$3+lr$4+decay$5+weight_+$6+noise$7warmup$8 \
    --epochs 60 \
    --constraint_lr $4 \
    --constraint_decay $5 \
    --lambda_constraint_weight $6 \
    --lambda_weight_mean 1 \
    --decrease_affine_lr 1 \
    --sample_noise True \
    --noise_data_dependent False \
    --noise_mean_std $7 \
    --noise_var_std $7 \
    --warmup_noise $8 \
    ${@:9}
