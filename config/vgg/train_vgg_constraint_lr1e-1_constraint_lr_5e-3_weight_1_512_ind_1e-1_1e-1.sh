python train_constraint_bn_v2_1.py --model vgg16_constraint_bn_v2 \
    --log_dir vgg/vgg16_constraint_lr1e-1_bsz_128_lr_5e-3_weight_1_512_noise_ind_1e-1_1e-1 \
    --lr 0.1 \
    --constraint_lr 0.005 \
    --constraint_decay 1 \
    --lambda_constraint_weight 0.001953 \
    --noise_data_dependent False \
    --noise_mean_std 0.1 \
    --noise_var_std 0.1 \
    --sample_noise True \
    --decrease_affine_lr 0.1 \


