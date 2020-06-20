python -m torch.distributed.launch --nproc_per_node=$1 \
    --master_port $2 \
    main_amp.py -a resnet_brn_18 \
    --b 256 \
    --sync_bn \
    --workers 2 \
    ./data/imagenet \
    --log_dir imagenet/resnet18_brn_bsz256_momentum1e-1
