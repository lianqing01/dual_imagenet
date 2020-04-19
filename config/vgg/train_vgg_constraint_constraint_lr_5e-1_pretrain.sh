python train_constraint_bn_v2.py --model vgg16_constraint_bn_v2 \
    --log_dir vgg/oracle_vgg16_constraint_constraint_lr_5e-1_pretrain \
    --lr 0.05 \
    --constraint_lr 0.5 \
    --initialize_by_pretrain \
    --constraint_decay 1 

