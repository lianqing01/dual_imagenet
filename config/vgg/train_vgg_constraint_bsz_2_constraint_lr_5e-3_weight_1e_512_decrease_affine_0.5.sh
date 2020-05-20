python train_constraint_bn_v2_1.py --model vgg16_constraint_bn_v2 \
    --log_dir vgg/vgg16_constraint_bsz_2_lr_5e-3_weight_1_512_decrease_affine_0.5 \
    --lr 0.005 \
    --constraint_lr 0.005 \
    --batch-size 2 \
    --constraint_decay 1 \
    --lambda_constraint_weight 0.001953 \
    --decrease_affine_lr 0.5 \


