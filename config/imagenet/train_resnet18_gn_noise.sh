python -m torch.distributed.launch --nproc_per_node=$1 --master_port 5555 main_amp.py -a resnet18 --b 256 --workers 8 --opt-level O2  ./data/imagenet \
    --sample_noise True \
    --noise_mean_std 0.05 \
    --noise_var_std 0.05 \
    --log_dir resnet18_gn_noise_ind_0.05
