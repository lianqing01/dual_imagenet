python train_constraint_bn_v2_1.py --model vgg16_constraint_bn_v2 \
    --log_dir vgg/oracle_vgg16_bsz_8_constraint_constraint_lr_5e-5_weight_1_512_fix \
    --lr 0.005 \
    --batch-size 8 \
    --constraint_lr 0.00005 \
    --constraint_decay 1 \
    --lambda_constraint_weight 0.001953

