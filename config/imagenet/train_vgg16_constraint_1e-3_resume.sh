python -m torch.distributed.launch --nproc_per_node=$1 --master_port=3622 main_amp_constraint_resume.py -a vgg_imagenet16_constraint --b 512 --workers 8 --opt-level O2  \
    ./data/imagenet \
    --log_dir imagenet/vgg_imagenet16_constraint_1e-3 \
    --lr 0.1 \
    --constraint_lr 0.005 \
    --constraint_decay 1 \
    --lambda_constraint_weight 1e-3 \
    --resume results/imagenet/vgg_imagenet16_constraint_1e-3_Sat-Jun--9-21:40:17-2020/74_checkpoint.pth.tar \
    --lambda_weight_mean 5 \
    --decrease_affine_lr 1 \
    --epochs 90 \
    --sample_noise False \
    --noise_data_dependent False \
    --noise_mean_std 0.05 \
    --noise_var_std 0.05 \

