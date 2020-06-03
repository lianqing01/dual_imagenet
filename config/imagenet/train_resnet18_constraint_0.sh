python -m torch.distributed.launch --nproc_per_node=$1 --master_port=3222 main_amp_constraint.py -a resnet_constraint18 --b 512 --workers 8 --opt-level O1  \
    ./data/imagenet \
    --log_dir imagenet/constraint_18_1e-3 \
    --constraint_lr 0.01 \
    --constraint_decay 1 \
    --lambda_constraint_weight 1e-3 \
    --lambda_weight_mean 1 \
    --decrease_affine_lr 1 \
    --sample_noise False \
    --noise_data_dependent False \
    --noise_mean_std 0.05 \
    --noise_var_std 0.05 \

