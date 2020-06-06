python -m torch.distributed.launch --nproc_per_node=$1 --master_port=34343 main_amp_constraint.py -a resnet_constraint_init18 --b 512 --workers 8 --opt-level O1  \
    ./data/imagenet \
    --log_dir imagenet/constraint_18_init_ind_5e-2 \
    --constraint_lr 0.01 \
    --constraint_decay 1 \
    --lambda_constraint_weight 1e-4 \
    --lambda_weight_mean 5 \
    --decrease_affine_lr 1 \
    --sample_noise True \
    --noise_data_dependent False \
    --noise_mean_std 0.05 \
    --noise_var_std 0.05 \

