python -m torch.distributed.launch --nproc_per_node=$1 --master_port=3622 main_amp_constraint.py -a vgg_imagenet16_constraint --b 512 --workers 8 --opt-level O2  \
    ./data/imagenet \
    --log_dir imagenet/vgg_imagenet16_constraint_1e-3 \
    --lr 0.1 \
    --constraint_lr 0.005 \
    --constraint_decay 1 \
    --lambda_constraint_weight 1e-3 \
    --lambda_weight_mean 5 \
    --decrease_affine_lr 1 \
    --sample_noise True \
    --noise_data_dependent False \
    --noise_mean_std 0.05 \
    --noise_var_std 0.05 \

