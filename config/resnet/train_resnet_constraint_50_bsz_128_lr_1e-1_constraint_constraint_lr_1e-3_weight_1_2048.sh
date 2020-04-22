python train_constraint_bn_v2_1.py --model resnet_constraint50 \
    --log_dir resnet/oracle_resnet_constraint_50_bsz_128_lr_1e-1_constraint_constraint_lr_1e-3_weight_1_2048_exp001 \
    --lr 0.1 \
    --batch-size 128 \
    --constraint_lr 0.001 \
    --constraint_decay 1 \
    --lambda_constraint_weight 0.000488

