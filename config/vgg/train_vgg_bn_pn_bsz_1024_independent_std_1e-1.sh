python train_pn.py --model vgg16_pn \
    --log_dir vgg/vgg16_pn_bsz_128_pn-bsz_1024_noise_independent_noise_std_1e-1 \
    --pn-batch-size 1024 \
    --batch-size 128 \
    --sample_noise True \
    --data_dependent False \
    --noise_std 0.1 \
    --batch_renorm True \
    --lr 0.05 \
