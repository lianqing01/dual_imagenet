python -m torch.distributed.launch --nproc_per_node=$1 \
    --master_port=$2 \
    main_amp_constraint.py -a resnet_constraint18 --b 256 \
    --workers 8 \
    ./data/tiny_imagenet \
    --mixed_precision False \
    --log_dir tiny/resnet18_constraint \
    --constraint_lr 0.01 \
    --constraint_decay 1 \
    --lambda_constraint_weight 1e-3 \
    --lambda_weight_mean 1 \
    --sample_noise False \
    --noise_data_dependent False \
    --noise_mean_std 0.05 \
    --noise_var_std 0.05 \

