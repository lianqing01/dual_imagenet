python train_pn.py --model resnet18 \
    --log_dir resnet/oracle_resnet18_bn_bsz_128_pn-bsz_256 \
    --batch-size 128 \
    --pn-batch-size 256 \
    --lr 0.1 \
    --dataset CIFAR100 \
    --project_name dual_bn_100
