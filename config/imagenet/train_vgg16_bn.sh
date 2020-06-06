python -m torch.distributed.launch --nproc_per_node=$1 main_amp.py -a vgg_imagenet16_bn --b 512 --workers 8 --opt-level O2  ./data/imagenet  \
    --log_dir vgg_imagenet16_bn \
