python train_constraint_bn_v2.py --model vgg16_constraint_bn_v2 \
    --log_dir vgg/oracle_vgg16_constraint_constraint_lr_5e-3_pretrain \
    --lr 0.05 \
    --constraint_lr 0.05 \
    --constraint_decay 1 \
    --initialize_by_pretrain \

