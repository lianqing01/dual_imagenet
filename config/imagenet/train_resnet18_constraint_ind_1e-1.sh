<<<<<<< HEAD
<<<<<<< HEAD
python -m torch.distributed.launch --nproc_per_node=$1 main_amp_constraint.py -a resnet_constraint18 --b 512 --workers 8 --opt-level O1  \
    ./data/imagenet --sync_bn \
    --log_dir imagenet/constraint_20_weight_1e-3_ind_1e-1 \
    --constraint_lr 0.005 \
    --constraint_decay 1 \
    --lambda_constraint_weight 0.001 \
=======
python -m torch.distributed.launch --nproc_per_node=$1 --master_port=3222 main_amp_constraint.py -a resnet_constraint18 --b 512 --workers 8 --opt-level O1  \
=======
python -m torch.distributed.launch --nproc_per_node=$1 --master_port=3122 main_amp_constraint.py -a resnet_constraint18 --b 512 --workers 8 --opt-level O1  \
>>>>>>> 0bd4c660cccd3b5a9943e3865ba98c02a4a553d9
    ./data/imagenet \
    --log_dir imagenet/constraint_18_1e-3_ind_1e-1 \
    --constraint_lr 0.01 \
    --constraint_decay 1 \
    --lambda_constraint_weight 1e-3 \
>>>>>>> 4c208638aa08c9888c6d89d7761af8e26f190898
    --lambda_weight_mean 1 \
    --decrease_affine_lr 1 \
    --sample_noise True \
    --noise_data_dependent False \
    --noise_mean_std 0.1 \
    --noise_var_std 0.1 \

