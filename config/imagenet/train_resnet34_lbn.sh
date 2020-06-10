python -m torch.distributed.launch --nproc_per_node=$1 --master_port 5555 main_amp.py -a resnet34 --b 1024 --workers 8 --opt-level O2  ./data/imagenet \
    --sync_bn \
    --log_dir resnet34_bn
