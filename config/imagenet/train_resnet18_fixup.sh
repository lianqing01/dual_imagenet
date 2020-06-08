python -m torch.distributed.launch --nproc_per_node=$1 main_amp.py -a fixup_resnet18 --b 512 --workers 8 --opt-level O1  ./data/imagenet
