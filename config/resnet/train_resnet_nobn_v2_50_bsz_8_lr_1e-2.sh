python train_constraint_bn_v2_1.py --model resnet_nobnv2_50 \
    --log_dir resnet/oracle_resnet_nobnv2_50_bsz_8_lr_1e-2 \
    --lr 0.01 \
    --batch-size 8 \
    --constraint_lr 0 \
    --lambda_constraint_weight 0

