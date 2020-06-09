python -m torch.distributed.launch --nproc_per_node=$1 --master_port 5555 main_amp.py -a resnet18 --b 256 --workers 8 --opt-level O1  ./data/tiny_imagenet \
    --log_dir tiny/resnet18_bn
