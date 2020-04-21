python train_constraint_bn_v2_1.py --model resnet_constraint50 \
    --log_dir resnet/oracle_resnet_constraint_50_bsz_128_constraint_constraint_lr_5e-4_weight_1_512_exp001 \
    --lr 0.005 \
    --batch-size 128 \
    --constraint_lr 0.0005 \
    --constraint_decay 1 \
    --lambda_constraint_weight 0.001953

