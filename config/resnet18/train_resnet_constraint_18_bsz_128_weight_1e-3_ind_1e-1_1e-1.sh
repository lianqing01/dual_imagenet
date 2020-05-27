python train_constraint_bn_v2_1.py --model resnet_constraint18 \
    --log_dir resnet/oracle_resnet_constraint_18_bsz_128_constraint_weight_1e-3_ind_1e-1_1e-1 \
    --lr 0.1 \
    --batch-size 128 \
    --constraint_lr 0.01 \
    --constraint_decay 1 \
    --lambda_constraint_weight 1e-3 \
    --sample_noise True \
    --noise_data_dependent False \
    --noise_mean_std 0.1 \
    --noise_var_std 0.1 \
    --decrease_affine_lr 0.1 

