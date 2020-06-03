python train_constraint_bn_v2_1.py --model vgg16_constraint_bn_v2 \
    --log_dir vgg/vgg16_constraint_lr1e-1_bsz_1024 \
    --lr 0.1 \
    --constraint_lr 0.01 \
    --constraint_decay 1 \
    --batch-size 1024 \
    --lambda_constraint_weight 0.001 \
    --noise_data_dependent False \
    --noise_mean_std 0.05 \
    --noise_var_std 0.05 \
    --sample_noise True \
    --decrease_affine_lr 0.1 \


