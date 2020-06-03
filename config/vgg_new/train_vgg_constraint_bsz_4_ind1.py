import subprocess
import argparse
import time
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--model', default="vgg16_constraint_bn_v2", type=str, help='learning rate')
parser.add_argument('--gpus', default=[0,1,2,3], type=list)
parser.add_argument('--noise', default=[1e-1, 1e-2], type=list)


args = parser.parse_args()

for i in range(len(args.noise)):
    script = "CUDA_VISIBLE_DEVICES={} python train_constraint_bn_v2_1.py \
    --model {} \
    --log_dir vgg/vgg16_constraint_bsz_4_noise_ind1_{} \
    --lr 0.005 \
    --constraint_lr 0.0005 \
    --batch-size 4 \
    --constraint_decay 1 \
    --lambda_constraint_weight 0.001 \
    --noise_data_dependent False \
    --noise_mean_std {} \
    --noise_var_std {} \
    --sample_noise True \
    --decrease_affine_lr 0.1".format(args.gpus[i], args.model, args.noise[i], args.noise[i], args.noise[i], args.noise[i])
    print(script)
    gpu_script = "export CUDA_VISIBLE_DEVICES={}".format(args.gpus[i])
    subprocess.Popen(gpu_script, shell=True)
    print(gpu_script)
    time.sleep(2)
    subprocess.Popen(script, stdin=None, stdout=None, stderr=None, shell=True)
