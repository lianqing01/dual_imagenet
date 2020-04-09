#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.
from __future__ import print_function

import argparse
import csv
import os

import numpy as np
from models.dual_norm import DualNorm
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import models
from torch.utils.tensorboard import SummaryWriter
from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--model', default="ResNet18", type=str,
                    help='model type (default: ResNet18)')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
parser.add_argument('--epoch', default=200, type=int,
                    help='total epochs to run')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='use standard augmentation (default: True)')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--log_dir', default="oracle_exp001")
parser.add_argument('--grad_clip', default=3)


# param for dual norm
parser.add_argument('--lamdba_dual_weight', default=1, type=int)
parser.add_argument('--dual_lr', default=0.1, type=float)
parser.add_argument('--dual_decay', default=1e-3, type=str)


args = parser.parse_args()
args.dual_decay = float(args.dual_decay)

use_cuda = torch.cuda.is_available()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if args.seed != 0:
    torch.manual_seed(args.seed)

# Data
print('==> Preparing data..')
if args.augment:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
else:
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = datasets.CIFAR10(root='~/data', train=True, download=False,
                            transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=args.batch_size,
                                          shuffle=True, num_workers=8)

testset = datasets.CIFAR10(root='~/data', train=False, download=False,
                           transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=8)


# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7' + args.name + '_'
                            + str(args.seed))
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    rng_state = checkpoint['rng_state']
    torch.set_rng_state(rng_state)
else:
    print('==> Building model..')
    net = models.__dict__[args.model]()

if not os.path.isdir('results/{}'.format(args.log_dir)):
    os.makedirs('results/{}'.format(args.log_dir))
logname = ('results/{}/log_'.format(args.log_dir) + net.__class__.__name__ + '_' + args.name + '_'
           + str(args.seed) + '.csv')

tb_logger = SummaryWriter(log_dir="results/{}".format(args.log_dir))

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net)
    print(torch.cuda.device_count())
    cudnn.benchmark = True
    print('Using CUDA..')

criterion = nn.CrossEntropyLoss()
print(args.lr)
dual_param = []
for m in net.modules():
    if isinstance(m, DualNorm):
        dual_param.extend(list(map(id, m.parameters())))
origin_param = filter(lambda p:id(p) not in dual_param, net.parameters())

optimizer = optim.SGD(origin_param, lr=args.lr, momentum=0.9,
                      weight_decay=args.decay)
dual_optimizer = (optim.SGD(
                    filter(lambda p:id(p) in dual_param, net.parameters()),
                    lr=args.dual_lr, momentum=0.9,
                    weight_decay=args.dual_decay
                    ))




def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    reg_loss = 0
    correct = 0
    total = 0
    mean = 0
    var = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        outputs = net(inputs)
        loss = criterion(outputs, targets)


        # dual loss
        weight_mean = 0
        weight_var = 0
        for m in net.modules():
            if isinstance(m, DualNorm):
                weight_mean_, weight_var_ =  m._get_weight_mean_var()
                weight_mean += weight_mean_
                weight_var += weight_var_

        dual_loss = weight_mean + weight_var
        dual_loss = -1 * args.lamdba_dual_weight * dual_loss

        # optimize dual loss

        dual_optimizer.zero_grad()
        dual_loss.backward(retain_graph=True)
        dual_optimizer.step()

        train_loss += loss.data.item()
        loss -= dual_loss

        # optimize
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
        optimizer.step()

        progress_bar(batch_idx, len(trainloader),
                     'Loss: %.3f | Dual Loss: %.4f | Mean %.4f | Var %.4f | Reg: %.5f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), dual_loss.item(), mean, var,
                        reg_loss/(batch_idx+1),
                        100.*correct/total, correct, total))
        if (batch_idx+1) % 10 == 0:
            mean = 0
            var = 0
            for m in net.modules():
                if isinstance(m, DualNorm):
                    mean_, var_ = m._get_mean_var()
                    mean += mean_.abs()
                    var += var_.abs()
            curr_idx = epoch * len(trainloader) + batch_idx
            tb_logger.add_scalar("train/train_loss", train_loss / (batch_idx+1), curr_idx)
            tb_logger.add_scalar("train/train_acc", 100.*correct/total, curr_idx)
            tb_logger.add_scalar("train/norm_mean", mean, curr_idx)
            tb_logger.add_scalar("train/norm_var", var, curr_idx)

    for m in net.modules():
        if isinstance(m, DualNorm):
            m._reset_mean_var()
    return (train_loss/batch_idx, reg_loss/batch_idx, 100.*correct/total)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss/(batch_idx+1), 100.*correct/total,
                        correct, total))
    acc = 100.*correct/total
    if epoch == start_epoch + args.epoch - 1 or acc > best_acc:
        checkpoint(acc, epoch)
    if acc > best_acc:
        best_acc = acc
    mean = 0
    var = 0
    for m in net.modules():
        if isinstance(m, DualNorm):
                mean_, var_ = m._get_mean_var()
                mean += mean_.abs()
                var += var_.abs()

    curr_idx = epoch * len(trainloader)
    tb_logger.add_scalar("test/test_loss", test_loss/batch_idx, curr_idx)
    tb_logger.add_scalar("test/test_acc", 100.*correct/total, curr_idx)
    tb_logger.add_scalar("test/norm_mean", mean, curr_idx)
    tb_logger.add_scalar("test/norm_var", var, curr_idx)
    for m in net.modules():
        if isinstance(m, DualNorm):
            m._reset_mean_var()

    return (test_loss/batch_idx, 100.*correct/total)


def checkpoint(acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint'):
        os.makedirs('checkpoint')
    torch.save(state, './checkpoint/ckpt.t7' + args.name + '_'
               + str(args.seed))


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = args.lr
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['epoch', 'train loss', 'reg loss', 'train acc',
                            'test loss', 'test acc'])

for epoch in range(start_epoch, args.epoch):
    train_loss, reg_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    adjust_learning_rate(optimizer, epoch)
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, train_loss, reg_loss, train_acc, test_loss,
                            test_acc])
