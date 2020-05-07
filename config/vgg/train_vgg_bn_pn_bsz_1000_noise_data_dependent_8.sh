python train_pn.py --model vgg16_pn \
    --log_dir vgg/vgg16_pn_bsz_128_pn-bsz_1000_noise_data_dependent_noise_bsz_8 \
    --pn-batch-size 1000 \
    --batch-size 128 \
    --sample_noise True \
    --data_dependent True \
    --noise_bsz 8 \
    --lr 0.05 \
