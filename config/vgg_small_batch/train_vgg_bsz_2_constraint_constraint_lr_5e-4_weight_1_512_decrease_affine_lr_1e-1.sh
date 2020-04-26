python train_constraint_bn_v2_1.py --model vgg16_constraint_bn_v2 \
    --log_dir vgg/oracle_vgg16_bsz_2_constraint_constraint_lr_5e-4_weight_1_512_fix_decrease_affine_lr_1e-1 \
    --lr 0.005 \
    --batch-size 2 \
    --constraint_lr 0.0005 \
    --constraint_decay 1 \
    --load_model  results/vgg/oracle_vgg16_bsz_2_constraint_constraint_lr_5e-4_weight_1_512_fix_decrease_affine_lr_1e-1/epoch_89.pth \
    --resume \
    --lambda_constraint_weight 0.001953 \
    --decrease_affine_lr 0.1

