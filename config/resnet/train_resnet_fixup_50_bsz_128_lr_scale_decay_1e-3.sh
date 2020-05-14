python train.py --model fixup_resnet50 \
    --log_dir resnet/oracle_resnet_fixup_50_bn_bsz_128_lr_scale_decay_1e-3 \
    --batch-size 128 \
    --fixup_scale_decay 0.001 \
    --lr 0.1 \
    --fixup True 
