python train_constraint_bn_v2_1.py --model resnet_constraint18 \
    --log_dir resnet/oracle_resnet_constraint_18_bsz_128_constraint_weight_1e-3_ind_2e-9_2e-9 \
    --lr 0.1 \
    --batch-size 128 \
    --constraint_lr 0.01 \
    --constraint_decay 1 \
    --lambda_constraint_weight 1e-3 \
    --lambda_weight_mean 10 \
    --sample_nosie True \
    --noise_data_dependent False \
    --sample_mean_std 0.09 \
    --sample_var_std 0.09 \
    --decrease_affine_lr 0.1 

