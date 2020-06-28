python -m torch.distributed.launch --nproc_per_node=$1 \
    --master_port $2 \
    main_amp.py -a resnet_brn_50 \
    --b 256 \
    --sync_bn \
    --workers 4 \
    --mixed_precision False \
    --opt-level O1 \
    ./data/imagenet \
    --log_dir imagenet/resnet50_brn_bsz256_momentum1e-1
