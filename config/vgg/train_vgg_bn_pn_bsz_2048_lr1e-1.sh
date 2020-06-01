python train_pn.py --model vgg16_pn \
    --log_dir vgg/vgg16_pn_bsz_128_pn-bsz_2048_lr1e-1 \
    --pn-batch-size 2048 \
    --batch-size 128 \
    --lr 0.1 \
    --sample_noise True \
    --data_dependent True \
