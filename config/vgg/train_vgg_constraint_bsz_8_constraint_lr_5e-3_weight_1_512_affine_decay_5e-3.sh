python train_constraint_bn_v2_1.py --model vgg16_constraint_bn_v2 \
    --log_dir vgg/vgg16_constraint_bsz_8_lr_5e-3_weight_1_512_affine_decay_5e-3 \
    --lr 0.005 \
    --constraint_lr 0.005 \
    --batch-size 8 \
    --constraint_decay 1 \
    --affine_weight_decay 0.005 \
    --lambda_constraint_weight 0.001953 \
    --decrease_affine_lr 0.1 \


