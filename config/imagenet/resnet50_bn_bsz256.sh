#!/usr/bin/env bash

python -m torch.distributed.launch --nproc_per_node=$1 \
    --master_port $2 \
    main_amp_finetune.py -a resnet50 \
    --b 256 \
    --workers 2 \
    --epochs 60 \
    --mixed_precision True \
    --opt-level O1 \
    ./data/imagenet \
    --log_dir finetune_resnet50_bn_bsz256+norm+$3+noise$4+warmup$5 \
    --norm_layer $3 \
    --sample_noise True \
    --noise_std_mean $4 \
    --noise_std_var $4 \
    --warmup_noise $5 \
    --pretrained \
    ${@:6}
