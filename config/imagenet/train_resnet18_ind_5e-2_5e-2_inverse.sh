python -m torch.distributed.launch --nproc_per_node=$1 --master_port 5555 main_amp.py -a resnet18 --b 1024 --workers 8 --opt-level O2  ./data/imagenet \
    --sync_bn \
    --sample_noise True \
    --noise_mean_std 0.01 \
    --noise_var_std 0.01 \
    --log_dir resnet18_bn_ind_5e-2_inverse
