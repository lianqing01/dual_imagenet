python train_constraint_bn_v2.py --model Two_layer \
    --log_dir two_layer_lr_1e-3_constraint_constraintlr_1e-5_constraint_decay_1e-2_exp001 \
    --lr 0.001 \
    --constraint_lr 0.00001 \
    --constraint_decay 0.01 \
    --two_layer
