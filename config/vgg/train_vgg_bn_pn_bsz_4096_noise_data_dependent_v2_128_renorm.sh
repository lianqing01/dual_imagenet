python train_pn.py --model vgg16_pn \
    --log_dir vgg/vgg16_pn_bsz_128_pn-bsz_4096_noise_data_dependent_v2_noise_bsz_128_batch_renorm \
    --pn-batch-size 2048 \
    --batch-size 128 \
    --sample_noise True \
    --r_max 6 \
    --batch_renorm True \
    --data_dependent True \
    --noise_bsz 128 \
    --lr 0.05 \
