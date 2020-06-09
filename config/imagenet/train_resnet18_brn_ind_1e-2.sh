python -m torch.distributed.launch --nproc_per_node=$1 --master_port 5555 main_amp.py -a resnet_brn_18 --b 512 --workers 8 --opt-level O2  ./data/imagenet \
    --log_dir resnet18_brn_ind_1e-2 \
    --sample_noise True \
    --noise_mean_std 0.01 \
    --noise_var_std 0.01 \
