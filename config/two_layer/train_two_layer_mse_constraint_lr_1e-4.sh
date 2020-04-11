python train_constraint_bn_v2.py --model Two_layer \
    --log_dir two_layer_mse_constraint_lr_1e-4_exp001 \
    --lr 0.0001 \
    --optim_loss mse \
    --num_classes 1 \
    --constraint_lr 0.0001 \
    --constraint_decay 0.01 \
    --two_layer
