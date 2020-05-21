python train_pn.py --model resnet18 \
    --log_dir resnet/oracle_resnet18_bn_bsz_128_pn-bsz_1024 \
    --batch-size 128 \
    --pn-batch-size 1024 \
    --lr 0.1 \
    --project_name dual_bn_100
