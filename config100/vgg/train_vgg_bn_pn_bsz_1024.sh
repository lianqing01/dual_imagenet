python train_pn.py --model vgg16_bn \
    --log_dir vgg/cifar100_vgg16_pn_bsz_128_pn-bsz_1024 \
    --pn-batch-size 1024 \
    --batch-size 128 \
    --dataset CIFAR100 \
    --lr 0.1 \
    --project_name dual_bn_100
