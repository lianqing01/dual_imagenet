python -m torch.distributed.launch --nproc_per_node=$1 --master_port=$2 main_amp_constraint.py -a resnet_constraint50 --b 256 --workers 4  \
    --mixed_precision False \
    ./data/imagenet  \
    --norm_layer $3 \
    --log_dir imagenet/constraint_20+norm_layer_+$3+noise_0.0001+wamrup_416181 \
    --epochs 120 \
    --constraint_lr 0.01 \
    --constraint_decay 0.5 \
    --lambda_constraint_weight 1 \
    --lambda_weight_mean 1 \
    --decrease_affine_lr 1 \
    --sample_noise False \
    --noise_data_dependent True \
    --noise_mean_std 0.0001 \
    --noise_var_std 0.0001 \
    --warmup_noise 21,41,61

