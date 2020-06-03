python -m torch.distributed.launch --nproc_per_node=$1 --master_port 1222 main_amp_constraint.py -a resnet_constraint18 --b 512 --workers 4 --opt-level O1  \
    ./data/imagenet \
    --log_dir imagenet/constraint_20_ind_1e-1_1e-1 \
    --constraint_lr 0.005 \
    --constraint_decay 1 \
    --lambda_constraint_weight 1e-2 \
    --lambda_weight_mean 1 \
    --decrease_affine_lr 0.1 \
    --sample_noise True \
    --noise_data_dependent False \
    --noise_mean_std 0.1 \
    --noise_var_std 0.1 \

