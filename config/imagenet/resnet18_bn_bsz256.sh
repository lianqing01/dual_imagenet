#!/usr/bin/env bash

python -m torch.distributed.launch --nproc_per_node=$1 \
    --master_port $2 \
    main_amp.py -a resnet18 \
    --b 512 \
    --workers 4 \
    --epochs 100 \
    --mixed_precision False \
    ./data/imagenet \
    --log_dir resnet18_bn_bsz512+norm+$3+noise$4+warmup$5 \
    --norm_layer $3 \
    --sample_noise True \
    --noise_std_mean $4 \
    --noise_std_var $4 \
    --warmup_noise $5 \
