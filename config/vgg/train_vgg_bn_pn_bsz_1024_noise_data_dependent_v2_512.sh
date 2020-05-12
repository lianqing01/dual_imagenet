python train_pn.py --model vgg16_pn \
    --log_dir vgg/vgg16_pn_bsz_128_pn-bsz_1024_noise_data_dependent_v2_noise_bsz_512_batch_renorm \
    --pn-batch-size 1024 \
    --batch-size 128 \
    --sample_noise True \
    --r_max 0.9 \
    --batch_renorm True \
    --data_dependent True \
    --noise_bsz 512 \
    --lr 0.05 \
