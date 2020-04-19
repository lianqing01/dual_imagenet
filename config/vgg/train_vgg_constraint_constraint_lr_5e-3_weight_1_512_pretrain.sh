python train_constraint_bn_v2.py --model vgg16_constraint_bn_v2 \
    --log_dir vgg/oracle_vgg16_constraint_constraint_lr_5e-3_weight_1_512_pretrain \
    --lr 0.05 \
    --constraint_lr 0.005 \
    --constraint_decay 1 \
    --lambda_constraint_weight 0.001953 \
    --initialize_by_pretrain \
    --max_pretrain_epoch 5

