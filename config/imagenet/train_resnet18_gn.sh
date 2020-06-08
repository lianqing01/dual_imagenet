python -m torch.distributed.launch --nproc_per_node=$1 --master_port 6555 main_amp.py -a resnet18 --b 256 --workers 6 --opt-level O2  ./data/imagenet \
    --log_dir resnet18_gn
