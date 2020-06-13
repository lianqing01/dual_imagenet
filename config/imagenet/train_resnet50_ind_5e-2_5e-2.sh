python -m torch.distributed.launch --nproc_per_node=$1 --master_port 5555 main_amp.py -a resnet50 --b 512 --workers 8 --opt-level O2  ./data/imagenet \
    --sync_bn \
    --sample_noise True \
    --noise_mean_std 0.001 \
    --noise_var_std 0.001 \
    --log_dir resnet50_lbn+noise
