python train_constraint_bn_v2_1.py --model vgg16_constraint_bn_v2 \
    --log_dir vgg/vgg16_constraint_bsz_128_lr_5e-3_weight_1_512_data_dependent \
    --lr 0.05 \
    --constraint_lr 0.005 \
    --constraint_decay 1 \
    --sample_noise False \
    --noise_data_dependent True \
    --resume \
    --load_model results/vgg/vgg16_constraint_bsz_128_lr_5e-3_weight_1_512_data_dependent_scratch/epoch_99.pth\
    --noise_std 0.01 \
    --get_norm_freq 100 \
    --lambda_constraint_weight 0.001953 \
    --decrease_affine_lr 0.1 \


