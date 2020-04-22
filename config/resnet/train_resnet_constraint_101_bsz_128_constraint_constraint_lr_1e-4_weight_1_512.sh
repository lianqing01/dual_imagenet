python train_constraint_bn_v2_1.py --model resnet_constraint101 \
    --log_dir resnet/oracle_resnet_constraint_101_bsz_128_constraint_constraint_lr_1e-4_weight_1_512_exp001 \
    --lr 0.1 \
    --batch-size 128 \
    --constraint_lr 0.0001 \
    --constraint_decay 1 \
    --lambda_constraint_weight 0.001953

