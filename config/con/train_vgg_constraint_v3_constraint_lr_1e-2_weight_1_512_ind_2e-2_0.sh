python train_constraint_bn_v2_1.py --model vgg16_constraint_bn_v3 \
    --log_dir vgg/vgg16_constraint_v3_bsz_128_lr_1e-2_weight_1_512_ind_2e-2 \
    --lr 0.1 \
    --constraint_lr 0.01 \
    --constraint_decay 1 \
    --lambda_constraint_weight 0.001953 \
    --decrease_affine_lr 0.1 \
    --noise_data_dependent False \
    --sample_noise True \
    --noise_mean_std 2e-2 \
    --noise_var_std 0 \


