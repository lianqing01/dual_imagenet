python train_constraint_bn_v2_1.py --model vgg16_constraint_bn_v2 \
    --log_dir vgg/smoothness \
    --lr 0.3 \
    --constraint_lr 0.015 \
    --constraint_decay 1 \
    --lambda_constraint_weight 0.001 \
    --noise_data_dependent False \
    --noise_mean_std 0.1 \
    --noise_var_std 0.1 \
    --sample_noise True \
    --lambda_weight_mean 1 \
    --decrease_affine_lr 0.1 \


