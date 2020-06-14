python -m torch.distributed.launch --nproc_per_node=$1 \
    --master_port $2 \
    main_amp.py -a down_resnet18 \
    --b 256 \
    --sync_bn \
    --workers 8 \
    --mixed_precision False \
    ./data/tiny_imagenet \
    --log_dir tiny/resnet18_bn_down_bsz256
