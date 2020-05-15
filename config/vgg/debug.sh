python train_constraint_bn_v2_1.py --model vgg16_constraint_bn_v2 \
    --log_dir vgg/debug \
    --lr 0.05 \
    --constraint_lr 0.005 \
    --constraint_decay 1 \
    --sample_noise False \
    --noise_data_dependent True \
    --noise_std 0.01 \
    --resume \
    --load_model results/vgg/debug/epoch_99.pth \
    --lambda_constraint_weight 0.001953 \
    --decrease_affine_lr 0.1 \


