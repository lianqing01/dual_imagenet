python -m torch.distributed.launch --nproc_per_node=$1 main_amp.py -a resnet50 --b 128 --workers 8 --opt-level O1  ./data/imagenet --sync_bn
