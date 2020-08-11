python -m torch.distributed.launch --nproc_per_node=$1 \
    --master_port $2 \
    main_amp.py -a resnet18 \
    --b 512 \
    --sync_bn \
    --workers 6 \
    --epochs 120 \
    --mixed_precision False \
    ./data/imagenet \
    --log_dir resnet18_bn_bsz512+norm+$3 \
    --norm_layer $3
