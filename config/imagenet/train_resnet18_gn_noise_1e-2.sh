python -m torch.distributed.launch --nproc_per_node=$1 --master_port 4444 main_amp.py -a resnet18 --b 512 --workers 8 --opt-level O2  ./data/imagenet \
    --sample_noise True \
    --noise_mean_std 0.01 \
    --noise_var_std 0.01 \
    --log_dir resnet18_gn_noise_ind_0.01
