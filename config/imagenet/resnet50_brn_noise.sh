python -m torch.distributed.launch --nproc_per_node=$1 \
    --master_port $2 \
    main_amp.py -a resnet_brn_50 \
    --b 128 \
    --sync_bn \
    --workers 4 \
    ./data/imagenet \
    --log_dir imagenet/resnet50_brn_bsz256_noise+$3 \
    --mixed_precision True \
    --sample_noise True \
    --noise_std_mean $3 \
    --noise_std_var $3 \
