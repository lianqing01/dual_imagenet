python train_constraint_bn_v2_1.py --model vgg16_constraint_bn_v2 \
    --log_dir vgg/oracle_vgg16_constraint_constraint_lr_0_decrease_affine_lr_10 \
    --lr 0.05 \
    --constraint_lr 0 \
    --constraint_decay 1 \
    --lambda_constraint_weight 0 \
    --decrease_affine_lr 0.1 \


