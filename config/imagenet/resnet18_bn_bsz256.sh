python -m torch.distributed.launch --nproc_per_node=$1 \
    --master_port $2 \
    main_amp.py -a resnet18 \
    --b 256 \
    --sync_bn \
    --workers 2 \
    --mixed_precision False \
    ./data/imagenet \
    --log_dir imagenet/resnet18_bn_bsz256
