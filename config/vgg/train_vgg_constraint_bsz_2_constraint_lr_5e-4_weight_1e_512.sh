python train_constraint_bn_v2_1.py --model vgg16_constraint_bn_v2 \
    --log_dir vgg/vgg16_constraint_bsz_2_lr_5e-4_weight_1_512 \
    --lr 0.005 \
    --constraint_lr 0.0005 \
    --batch-size 2 \
    --constraint_decay 1 \
    --lambda_constraint_weight 0.001953 \
    --decrease_affine_lr 0.1 \


