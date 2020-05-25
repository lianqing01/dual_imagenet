python train_constraint_bn_v2_1.py --model vgg16_constraint_bn_v2 \
    --log_dir vgg/vgg16_constraint_lr1e-1_bsz_128_lr_5e-3_weight_1_512_noise_after_x \
    --lr 0.1 \
    --constraint_lr 0.005 \
    --constraint_decay 1 \
    --add_noise after_x \
    --lambda_constraint_weight 0.001953 \
    --decrease_affine_lr 0.1 \


