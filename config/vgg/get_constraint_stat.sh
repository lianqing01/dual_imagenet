python get_constraint_stat.py --model vgg16_constraint_bn_v2 \
    --log_dir vgg/vgg16_constraint_bsz_128_norm_stat \
    --lr 0.05 \
    --constraint_lr 0.005 \
    --constraint_decay 1 \
    --resume \
    --load_model results/vgg/vgg16_bsz_128_constraint_lr_5e-3_weight_1_512/epoch_89.pth \
    --lambda_constraint_weight 0.001953 \
    --decrease_affine_lr 0.1 \


