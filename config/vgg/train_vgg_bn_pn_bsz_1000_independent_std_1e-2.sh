python train_pn.py --model vgg16_pn \
    --log_dir vgg/vgg16_pn_bsz_128_pn-bsz_1000_noise_dependent_noise_std_1e-2 \
    --pn-batch-size 1000 \
    --batch-size 128 \
    --sample_noise True \
    --data_dependent False \
    --noise_std 0.01 \
    --noise_bsz 32 \
    --lr 0.05 \
