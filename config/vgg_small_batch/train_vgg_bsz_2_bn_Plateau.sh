python train.py --model vgg16_bn \
    --log_dir vgg/oracle_vgg16_bsz_2_bn_Plateau \
    --batch-size 2 \
    --lr 0.005 \
    --lr_ReduceLROnPlateau True
