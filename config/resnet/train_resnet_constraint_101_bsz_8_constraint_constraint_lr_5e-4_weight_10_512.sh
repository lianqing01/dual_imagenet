python train_constraint_bn_v2.py --model resnet_constraint101 \
    --log_dir resnet/oracle_resnet_constraint_101_bsz_8_constraint_constraint_lr_5e-4_weight_10_512_exp001 \
    --lr 0.005 \
    --batch-size 8 \
    --constraint_lr 0.0005 \
    --constraint_decay 1 \
    --lambda_constraint_weight 0.01953

