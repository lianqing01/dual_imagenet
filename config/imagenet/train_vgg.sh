python -m torch.distributed.launch --nproc_per_node=$1 main_amp.py -a vgg16_bn --b 224 --workers 8 --opt-level O2 --keep-batchnorm-fp32 True ./data/imagenet --sync_bn
