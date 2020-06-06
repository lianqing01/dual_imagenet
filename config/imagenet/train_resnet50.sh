python -m torch.distributed.launch --nproc_per_node=$1 --master_port 6555 main_amp.py -a resnet50 --b 512 --workers 8 --opt-level O1  ./data/imagenet \
    --log_dir resnet18_bn
