python train_constraint_bn_v2_1.py --model vgg16_constraint_bn_v2 \
    --log_dir vgg/resume_vgg16_constraint_constraint_lr_5e-4_weight_1_512 \
    --lr 0.05 \
    --constraint_lr 0.0005 \
    --constraint_decay 1 \
    --lambda_constraint_weight 0.001953 \
    --resume \
    --load_model ./results/vgg/oracle_vgg16_constraint_constraint_lr_5e-3_weight_1_512_fix/epoch_99.pth

