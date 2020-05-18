python train_constraint_bn_v2_1.py --model vgg16_constraint_bn_v2 \
    --log_dir vgg/vgg16_lr_1e-1_constraint_bsz_128_lr_5e-3_weight_1_512 \
    --lr 0.5 \
    --constraint_lr 0.05 \
    --constraint_decay 1 \
    --lambda_constraint_weight 0.001953 \
    --decrease_affine_lr 0.1 \


