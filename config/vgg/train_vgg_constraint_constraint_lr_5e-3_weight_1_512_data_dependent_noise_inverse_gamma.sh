python train_constraint_bn_v2_1.py --model vgg16_constraint_bn_v2 \
    --log_dir vgg/vgg16_constraint_bsz_8_lr_5e-3_weight_1_512_sample_noise_1e-2_inverse_gamma \
    --lr 0.005 \
    --resume \
    --load_model results/vgg/vgg16_constraint_bsz_8/epoch_99.pth \
    --constraint_lr 0.005 \
    --constraint_decay 1 \
    --batch-size 8 \
    --sample_noise False \
    --noise_data_dependent True \
    --noise_std 0.01 \
    --lambda_constraint_weight 0.001953 \
    --decrease_affine_lr 0.1 \


