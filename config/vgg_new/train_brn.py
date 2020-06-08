import subprocess
import argparse
import time
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--model', default="vgg16_brn", type=str, help='learning rate')
parser.add_argument('--gpus', default=[0,1,2,3, 0, 1, 2, 3], type=list)
#parser.add_argument('--noise', default=[5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4], type=list)
parser.add_argument('--noise', default=[1e-1], type=list)


args = parser.parse_args()

for i in range(len(args.noise)):
    script = "CUDA_VISIBLE_DEVICES={} python train_pn.py \
    --model {} \
    --log_dir vgg/vgg16_brn_bsz_128_noise_ind_{} \
    --lr 0.1 \
    --batch-size 128 \
    --sample_noise True \
    --sample_std_mean {} \
    --sample_std_var {}".format(args.gpus[i], args.model, args.noise[i], args.noise[i], args.noise[i])
    print(script)
    gpu_script = "export CUDA_VISIBLE_DEVICES={}".format(args.gpus[i])
    subprocess.Popen(gpu_script, shell=True)
    print(gpu_script)
    time.sleep(2)
    subprocess.Popen(script, stdin=None, stdout=None, stderr=None, shell=True)
