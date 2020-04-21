python train_constraint_bn_v2_1.py --model resnet_constraint50 \
    --log_dir resnet/oracle_resnet_constraint_50_bsz_8_constraint_constraint_lr_1e-2_weight_1_2048_fix \
    --lr 0.01 \
    --batch-size 8 \
    --constraint_lr 0.0001 \
    --constraint_decay 1 \
    --lambda_constraint_weight 0.000488
