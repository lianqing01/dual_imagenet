python -m torch.distributed.launch --nproc_per_node=$1 --master_port=$2 main_amp_constraint.py -a resnet_constraint50 --b 256 --workers 4  \
    --mixed_precision True \
    --opt-level O1 \
    ./data/imagenet  \
    --norm_layer $3 \
    --log_dir imagenet/constraint_20+norm_layer_+$3 \
    --epochs 120 \
    --constraint_lr 0.005 \
    --constraint_decay 1 \
    --lambda_constraint_weight 1 \
    --lambda_weight_mean 1 \
    --decrease_affine_lr 1 \
    --sample_noise False \
    --noise_data_dependent False \
    --noise_mean_std 0 \
    --noise_var_std 0 \

