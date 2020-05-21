python train_pn.py --model resnet18 \
    --log_dir resnet/oracle_resnet18_bn_bsz_128_pn-bsz_2048 \
    --batch-size 128 \
    --pn-batch-size 2048 \
    --lr 0.1 \
    --project_name dual_bn_100
