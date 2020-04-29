python train_constraint_bn_v2_1.py --model vgg16_constraint_bn_v2 \
    --log_dir vgg/vgg16_constraint_bsz_2_conv_lr_0.0078125_constraint_lr_0 \
    --lr 0.00078125 \
    --batch-size 2 \
    --constraint_lr 0 \
    --constraint_decay 0 \
    --lambda_constraint_weight 0 \
    --decrease_affine_lr 0.1 \


