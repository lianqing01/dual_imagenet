python train_constraint_bn_v2_1.py --model vgg50_constraint_bn_v2 \
    --log_dir vgg/vgg50_constraint_bsz_128_lr_5e-3_weight_debug1 \
    --lr 0.05 \
    --constraint_lr 0.05 \
    --constraint_decay 1 \
    --lambda_constraint_weight 0.0001 \
    --decrease_affine_lr 0.1 \


