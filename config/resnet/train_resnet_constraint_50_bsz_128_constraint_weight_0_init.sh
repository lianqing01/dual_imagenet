python train_constraint_bn_v2_1.py --model resnet_constraint_init50 \
    --log_dir resnet/oracle_resnet_constraint_50_bsz_128_constraint_weight_0_init \
    --lr 0.1 \
    --batch-size 128 \
    --constraint_lr 0 \
    --constraint_decay 0 \
    --lambda_constraint_weight 0 \
    --decrease_affine_lr 0.1 

