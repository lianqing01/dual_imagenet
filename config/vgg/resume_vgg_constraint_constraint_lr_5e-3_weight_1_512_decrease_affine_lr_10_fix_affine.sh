python train_constraint_bn_v2_1.py --model vgg16_constraint_bn_v2 \
    --log_dir vgg/resume_vgg16_constraint_constraint_lr_5e-3_weight_1_512_decrease_affine_lr_10_fix_affine \
    --lr 0.05 \
    --constraint_lr 0.005 \
    --constraint_decay 1 \
    --lambda_constraint_weight 0.001953 \
    --decrease_affine_lr 0.1 \
    --load_model results/vgg/oracle_vgg16_constraint_constraint_lr_5e-3_weight_1_512_decrease_affine_lr_10/epoch_99.pth \
    --resume \
    --update_affine_only True

