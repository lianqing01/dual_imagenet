python train_constraint_bn_v2_1.py --model resnet_constraint18 \
    --log_dir resnet/oracle_resnet_constraint_18_bsz_128_constraint_weight_5e-3 \
    --lr 0.1 \
    --batch-size 128 \
    --constraint_lr 0.01 \
    --constraint_decay 0 \
    --lambda_constraint_weight 5e-3 \
    --decrease_affine_lr 0.1 

