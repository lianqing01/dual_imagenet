python train_constraint_bn_v2.py --model Two_layer \
    --log_dir two_layer_mse_constraint_optimal_constraint_decay_1e-3_exp001 \
    --lr 0.0001 \
    --optim_loss mse \
    --batch-size 1024 \
    --num_classes 1 \
    --constraint_lr 0.00001 \
    --constraint_decay 0.001 \
    --get_optimal_lagrangian \
    --two_layer
