python train_pn.py --model vgg16_pn \
    --log_dir vgg/vgg16_pn_bsz_128_pn-bsz_1024_noise_data_dependent_v2_noise_bsz_32 \
    --pn-batch-size 1024 \
    --batch-size 128 \
    --sample_noise True \
    --data_dependent True \
    --batch_renorm True \
    --r_max 0.9 \
    --noise_bsz 32 \
    --lr 0.1 \
