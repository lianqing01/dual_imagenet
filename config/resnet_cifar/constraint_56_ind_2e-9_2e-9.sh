python train_constraint_bn_v2_1.py --model resnet_constraint56 \
    --log_dir resnet/oracle_resnet_constraint_56_bsz_128_ind_2e-9_2e-9 \
    --lr 0.1 \
    --batch-size 128 \
    --constraint_lr 0.01 \
    --constraint_decay 1 \
    --lambda_constraint_weight 1e-4 \
    --lambda_weight_mean 10 \
    --decrease_affine_lr 0.1 \
    --sample_noise True \
    --noise_data_dependent False \
    --noise_mean_std 0.09 \
    --noise_var_std 0.09 \

