python -m torch.distributed.launch --nproc_per_node=$1 --master_port=$2 main_amp_constraint.py -a resnet_constraint18 --b 512  --workers 4  \
    ./data/imagenet  \
    --norm_layer $3 \
    --mixed_precision True \
    --opt-level O1 \
    --log_dir imagenet/constraint_20+norm_layer_+$3+noise_$4+warmup_+$5_cweight$6+cdecay$7 \
    --epochs 120 \
    --constraint_lr 0.01 \
    --constraint_decay $7 \
    --lambda_constraint_weight $6 \
    --lambda_weight_mean 1 \
    --decrease_affine_lr 1 \
    --sample_noise True \
    --noise_data_dependent False \
    --noise_mean_std $4 \
    --noise_var_std $4 \
    --warmup_noise $5 \
    --resume $8 \
    --lag_rho $9 \

