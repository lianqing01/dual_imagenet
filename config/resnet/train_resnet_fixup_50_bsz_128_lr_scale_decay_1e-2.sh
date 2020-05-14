python train.py --model fixup_resnet50 \
    --log_dir resnet/oracle_resnet_fixup_50_bn_bsz_128_lr_scale_decay_1e-2 \
    --batch-size 128 \
    --fixup_scale_decay 0.01 \
    --lr 0.1 \
    --fixup True 
