import subprocess
import argparse
import time
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--model', default="resnet_gn20", type=str, help='learning rate')
parser.add_argument('--gpus', default=[0,1,2,3], type=list)
parser.add_argument('--noise', default=[1e-1, 1e-2, 1e-3, 5e-2], type=list)


args = parser.parse_args()

for i in range(len(args.noise)):
    script = "CUDA_VISIBLE_DEVICES={} python train_pn.py \
    --model {} \
    --log_dir vgg/resnet20_gn_bsz_128_noise_ind_{} \
    --lr 0.1 \
    --batch-size 128 \
    --noise_mean_std {} \
    --noise_var_std {} \
    --grad_clip 1 \
    --sample_noise True".format(args.gpus[i], args.model, args.noise[i], args.noise[i], args.noise[i])
    print(script)
    gpu_script = "export CUDA_VISIBLE_DEVICES={}".format(args.gpus[i])
    subprocess.Popen(gpu_script, shell=True)
    print(gpu_script)
    time.sleep(2)
    subprocess.Popen(script, stdin=None, stdout=None, stderr=None, shell=True)
