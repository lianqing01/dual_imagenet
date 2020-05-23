import os
import numpy as np
for i in np.arange(0.1, 10, 0.1):
    script = "python train_constraint_bn_v2_1.py --model vgg16_constraint_bn_v2    --log_dir vgg/tune_{}    --lr 0.1    --resume    --load_model results/vgg/vgg16_constraint_lr1e-1_bsz_128_lr_5e-3_weight_1_512/epoch_99.pth    --epoch 120    --constraint_lr 0.005    --constraint_decay 1    --sample_noise False    --noise_data_dependent True    --noise_std 0.01    --get_norm_freq 100    --lambda_noise_weight {}    --lambda_constraint_weight 0.001953    --decrease_affine_lr 0.1 ".format(i, i)
    os.system(script)
    print(script)
