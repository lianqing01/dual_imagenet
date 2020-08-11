python -m torch.distributed.launch --nproc_per_node=$1 --master_port=$2 main_amp_constraint.py -a $4 --b 512 --workers 6  \
    ./data/imagenet  \
    --norm_layer $3 \
    --opt-level O1 \
    --mixed_precision True \
    --log_dir imagenet/constraint_20+norm_layer_+$3+model+$4+clr+$5+cdecay+$6+cweight+$7\
    --epochs 120 \
    --constraint_lr $5 \
    --constraint_decay $6 \
    --lambda_constraint_weight $7 \
    --lambda_weight_mean 1 \
    --decrease_affine_lr 1 \
    --sample_noise False \
    --noise_data_dependent False \
    --noise_mean_std 0 \
    --noise_var_std 0 \

