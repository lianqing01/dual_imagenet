python train_pn.py --model vgg16_pn \
    --log_dir vgg/vgg16_pn_bsz_128_pn-bsz_128_noise_independent_after_x_v1_mean_0_std_3e-1 \
    --pn-batch-size 128 \
    --batch-size 128 \
    --sample_noise True \
    --data_dependent after_x \
    --sample_mean_mean 0 \
    --sample_mean_var 0 \
    --sample_std_mean 1 \
    --sample_std_var 3e-1 \
    --batch_renorm False \
    --lr 0.1 \
