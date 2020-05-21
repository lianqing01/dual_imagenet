python train_constraint_bn_v2_1.py --model vgg16_constraint_bn_v2 \
    --log_dir vgg/visualize \
    --lr 0.1 \
    --constraint_lr 0.005 \
    --constraint_decay 1 \
    --sample_noise False \
    --noise_data_dependent True \
    --resume \
    --load_model results/vgg/debug/epoch_99.pth \
    --noise_std 0.01 \
    --lambda_constraint_weight 0.001953 \
    --decrease_affine_lr 0.1 \


