import argparse
import os
import shutil
import copy
import time

import wandb
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from models.constraint_bn_v2 import *
from utils import create_logger
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models
import math

from collections import OrderedDict

import torch.distributed as dist
from torch._utils import (_flatten_dense_tensors, _take_tensors,
                          _unflatten_dense_tensors)


def adjust_constraint_weight(args, curr_iter, max_iter):
    if args.max_lag_weight != 0:
        weight = args.lambda_constraint_weight + 1/2. * (args.max_lag_weight - args.lambda_constraint_weight) * \
                (1 + math.cos((max_iter - curr_iter) / max_iter * math.pi))
        return weight
    else:
        return args.lambda_constraint_weight

def _allreduce_coalesced(tensors, world_size, bucket_size_mb=-1):
    if bucket_size_mb > 0:
        bucket_size_bytes = bucket_size_mb * 1024 * 1024
        buckets = _take_tensors(tensors, bucket_size_bytes)
    else:
        buckets = OrderedDict()
        for tensor in tensors:
            tp = tensor.type()
            if tp not in buckets:
                buckets[tp] = []
            buckets[tp].append(tensor)
        buckets = buckets.values()

    for bucket in buckets:
        flat_tensors = _flatten_dense_tensors(bucket)
        dist.all_reduce(flat_tensors)
        flat_tensors.div_(world_size)
        for tensor, synced in zip(
                bucket, _unflatten_dense_tensors(flat_tensors, bucket)):
            tensor.copy_(synced)


def allreduce_params(params, coalesce=True, bucket_size_mb=-1):
    weights = [
        param.data for param in params
        if param.data is not None
    ]
    world_size = dist.get_world_size()
    if coalesce:
        _allreduce_coalesced(weights, world_size, bucket_size_mb)
    else:
        for tensor in weights:
            dist.all_reduce(tensor.div_(world_size))


def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')


import numpy as np
def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')



def fast_collate(batch, memory_format):

    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros( (len(imgs), 3, h, w), dtype=torch.uint8).contiguous(memory_format=memory_format)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        tensor[i] += torch.from_numpy(np.array(nump_array))
    return tensor, targets


def parse():
    model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(description='PyTorch Imagemodel Training')
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resmodel18',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resmodel18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size per process (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='Initial learning rate.  Will be scaled by <global batch size>/256: args.lr = args.lr*float(args.batch_size*args.world_size)/256.  A warmup schedule will also be applied over the first 5 epochs.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--grad_clip', default=1)

    parser.add_argument('--prof', default=-1, type=int,
                        help='Only run 10 iterations for profiling.')
    parser.add_argument('--deterministic', action='store_true')

    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--sync_bn', action='store_true',
                        help='enabling apex sync BN.')
    parser.add_argument('--optim_loss', default="cross_entropy")
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--print_freq', default=100, type=int)
    parser.add_argument('--mixed_precision', default=True, type=str2bool)
    parser.add_argument('--use_gc', default=False, type=str2bool)
    parser.add_argument('--constraint_weighted_average', default=True, type=str2bool)




    # param for constraint norm
    parser.add_argument('--lambda_constraint_weight', default=0, type=float)
    parser.add_argument('--constraint_lr', default=0.1, type=float)
    parser.add_argument('--constraint_decay', default=1e-3, type=float)
    parser.add_argument('--get_optimal_lagrangian',action='store_true', default=False)
    parser.add_argument('--decay_constraint', default=-1, type=int)
    parser.add_argument('--update_affine_only', default=False, type=str2bool)
    parser.add_argument('--lag_function', default=None, type=str)

    # two layer
    parser.add_argument('--two_layer', action='store_true', default=False)

    # for lr scheduler
    parser.add_argument('--lr_ReduceLROnPlateau', default=False, type=str2bool)
    parser.add_argument('--schedule', default=[100,150])
    parser.add_argument('--decrease_affine_lr', default=1, type=float)
    parser.add_argument('--decrease_with_conv_bias', default=False, type=str2bool)
    parser.add_argument('--affine_momentum', default=0.9, type=float)
    parser.add_argument('--affine_weight_decay', default=1e-4, type=float)

    # for adding noise
    parser.add_argument('--sample_noise', default=False, type=str2bool)
    parser.add_argument('--noise_data_dependent', default=False, type=str2bool)
    parser.add_argument('--noise_std', default=0, type=float)
    parser.add_argument('--lambda_noise_weight', default=1, type=float)
    parser.add_argument('--noise_mean_std', default=0, type=float)
    parser.add_argument('--noise_var_std', default=0, type=float)
    parser.add_argument('--norm_layer', default=None, type=str)
    parser.add_argument('--warmup_noise', default=None, type=str)
    parser.add_argument('--warmup_scale', default=10, type=float)
    parser.add_argument('--lag_rho', default=0, type=float)
    parser.add_argument('--warmup_roi', default=None, type=str)
    parser.add_argument('--max_lag_weight', default=0, type=float)


    # dataset
    parser.add_argument('--dataset', default='CIFAR10', type=str)
    parser.add_argument('--add_grad_noise', default=False, type=str2bool)
    parser.add_argument('--get_norm_freq', default=1, type=int)



    # pretrain
    parser.add_argument('--initialize_by_pretrain', action='store_true', default=False)
    parser.add_argument('--max_pretrain_epoch', default=20, type=int)
    parser.add_argument('--add_noise', default=None, type=str)
    parser.add_argument('--lambda_weight_mean', default=1, type=float)



    parser.add_argument('--opt-level', type=str)
    parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    parser.add_argument('--loss-scale', type=str, default=None)
    parser.add_argument('--channels-last', type=bool, default=False)
    parser.add_argument('--log_dir', default="", type=str)

    args = parser.parse_args()

    # For wamup noise
    if args.warmup_noise is not None:
        args.warmup_noise = args.warmup_noise.split(",")[:-1]
        args.warmup_noise = [int(i) for i in args.warmup_noise]
    if args.warmup_roi is not None:
        args.warmup_roi = args.warmup_roi.split(",")[:-1]
        args.warmup_roi = [int(i) for i in args.warmup_roi]



    return args

def main():
    global best_prec1, args

    args = parse()

    cudnn.benchmark = True
    best_prec1 = 0
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.local_rank)
        torch.set_printoptions(precision=10)

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) >= 1

    args.log_dir = args.log_dir + '_' + time.asctime(time.localtime(time.time())).replace(" ", "-")
    os.makedirs('results/{}'.format(args.log_dir), exist_ok=True)
    global logger
    logger = create_logger('global_logger', "results/{}/log.txt".format(args.log_dir))
    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        logger.info(args.local_rank)
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()
    logger.info(args.world_size)
    if args.local_rank == 0:

        wandb.init(project="tinyimagenet", dir="results/{}".format(args.log_dir),
           name=args.log_dir,)
        wandb.config.update(args)


        logger.info("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))




    args.batch_size = int(args.batch_size/args.world_size)
    logger.info(args.batch_size)

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    if args.channels_last:
        memory_format = torch.channels_last
    else:
        memory_format = torch.contiguous_format

    # create model

    import models
    logger.info('==> Building model..')
    global norm_layer
    print(args.norm_layer)
    if args.norm_layer is not None and args.norm_layer != 'False':
        if args.norm_layer == 'cbn':
            norm_layer =  models.__dict__['Constraint_Norm2d']
        elif args.norm_layer == 'cbn_mu_v1':
            norm_layer = models.__dict__['Constraint_Norm_mu_v1_2d']
        elif args.norm_layer == 'cbn_notheta':
            norm_layer = models.__dict__['Constraint_Norm_notheta_2d']
        else:
            norm_layer = None
        print(norm_layer)

    if args.pretrained:
        logger.info("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True, norm_layer=norm_layer)
    else:
        logger.info("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](norm_layer=norm_layer)

    model = model
    if args.lag_function is not None:
        if args.lag_function == 'lag_v1':
            lag_function = models.__dict__['LagrangianFunction_v1']
        elif args.lag_function == 'lag_v2':
            lag_function = models.__dict__['LagrangianFunction_v2']
        elif args.lag_function == 'lag_v3':
            lag_function = models.__dict__['LagrangianFunction_v3']
        elif args.lag_function == 'lag_v4':
            lag_function = models.__dict__['LagrangianFunction_v4']
        elif args.lag_function == 'lag_v5':
            lag_function = models.__dict__['LagrangianFunction_v5']
        for m in model.modules():
            if isinstance(m, Constraint_Lagrangian):
                m.lag_function = lag_function

    model = model.cuda()

    # Scale learning rate based on global batch size
    args.lr = args.lr*float(args.batch_size*args.world_size)/256.
    args.constraint_lr = args.constraint_lr*float(args.batch_size*args.world_size) / 256.
    constraint_param = []
    for m in model.modules():
        if isinstance(m, Constraint_Lagrangian):
            m.weight_decay = args.constraint_decay
            m.get_optimal_lagrangian = args.get_optimal_lagrangian
            constraint_param.extend(list(map(id, m.parameters())))


    if args.decrease_affine_lr == 1:
        origin_param = filter(lambda p:id(p) not in constraint_param, model.parameters())

        if args.use_gc:
            from algorithm.SGD import SGD
            SGD = SGD
            optimizer = SGD([
                        {'params': origin_param},
                        {'params':  filter(lambda p:id(p) in constraint_param, model.parameters()),
                                'lr': args.constraint_lr,
                                'weight_decay': args.constraint_decay},
                        ],
                        lr=args.lr, momentum=0.9,
                        weight_decay=args.decay, use_gc=True)
        else:
             optimizer = optim.SGD([
                        {'params': origin_param},
                        {'params':  filter(lambda p:id(p) in constraint_param, model.parameters()),
                                'lr': args.constraint_lr,
                                'weight_decay': args.constraint_decay},
                        ],
                        lr=args.lr, momentum=0.9,
                        weight_decay=args.decay)





    # Initialize Amp.  Amp accepts either values or strings for the optional override arguments,
    # for convenient interoperation with argparse.
    if args.mixed_precision:
        model, optimizer = amp.initialize(model, optimizer,
                                        opt_level=args.opt_level,
                                        keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                        loss_scale=args.loss_scale
                                        )

    # For distributed training, wrap the model with apex.parallel.DistributedDataParallel.
    # This must be done AFTER the call to amp.initialize.  If model = DDP(model) is called
    # before model, ... = amp.initialize(model, ...), the call to amp.initialize may alter
    # the types of model's parameters in a way that disrupts or destroys DDP's allreduce hooks.
    if args.distributed and args.mixed_precision:
        # By default, apex.parallel.DistributedDataParallel overlaps communication with
        # computation in the backward pass.
        model = DDP(model)
        # delay_allreduce delays all communication to the end of the backward pass.
    else:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # Optionally resume from a checkpoint
    if args.resume:
        # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(args.resume):
                logger.info("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(args.gpu))
                args.start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                logger.info("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                logger.info("=> no checkpoint found at '{}'".format(args.resume))
        resume()

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    if(args.arch == "inception_v3"):
        raise RuntimeError("Currently, inception_v3 is not supported by this example.")
        # crop_size = 299
        # val_size = 320 # I chose this value arbitrarily, we can adjust.
    else:
        crop_size = 224
        val_size = 256

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(), Too slow
            # normalize,
        ]))
    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(val_size),
            transforms.CenterCrop(crop_size),
        ]))

    train_sampler = None
    val_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    collate_fn = lambda b: fast_collate(b, memory_format)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        sampler=val_sampler,
        collate_fn=collate_fn)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    #initialization

    for m in model.modules():
        if isinstance(m, norm_layer):
            m.lagrangian.rho = args.lag_rho
    with torch.no_grad():
        if not args.resume:
            print("===initializtion====")
            _initialize(train_loader, model, criterion, optimizer, 0)
    device = torch.device("cuda")

    for m in model.modules():
        if isinstance(m, norm_layer):
            m.sample_noise = args.sample_noise
            m.sample_mean = torch.zeros(m.num_features).to(device)
            m.add_noise = args.add_noise
            m.sample_mean_std = torch.sqrt(torch.Tensor([args.noise_mean_std])[0].to(device))
            m.sample_var_std = torch.sqrt(torch.Tensor([args.noise_var_std])[0].to(device))


    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        if args.warmup_noise is not None:
            if epoch in args.warmup_noise:

                for m in model.modules():
                    if isinstance(m, norm_layer):
                        m.sample_mean_std *= math.sqrt(args.warmup_scale)
                        m.sample_var_std *= math.sqrt(args.warmup_scale)




        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(epoch, val_loader, model, criterion)


        # remember best prec@1 and save checkpoint
        if args.local_rank == 0:
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, filename = os.path.join("results/" + args.log_dir, "{}_checkpoint.pth.tar".format(epoch)))

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target

def _initialize(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    train_loss_avg = 0
    train_loss = AverageMeter()
    correct = 0
    total = 0
    mean = 0
    var = 0
    lambda_ = 0
    xi_ = 0


    # switch to train mode
    model.train()
    end = time.time()
    num_norm = 0
    for m in model.modules():
        if isinstance(m, norm_layer):
            m.reset_norm_statistics()
            num_norm+=1
    print("num_norm : {}".format(num_norm))

    prefetcher = data_prefetcher(train_loader)
    input, target = prefetcher.next()
    i = 0
    for layer in range(num_norm):
        for idx in range(2):
            while input is not None:
                i += 1
                if i>=11:
                    i = 0
                    break
                # compute output
                output = model(input)

                # compute gradient and do SGD step
                # constraint loss
                weight_mean = 0
                weight_var = 0
                for m in model.modules():
                    if isinstance(m, Constraint_Lagrangian):
                        weight_mean_, weight_var_ =  m.get_weight_mean_var()
                        weight_mean += weight_mean_
                        weight_var += weight_var_

                constraint_loss = args.lambda_weight_mean * weight_mean + weight_var
                constraint_loss = args.lambda_constraint_weight * constraint_loss

                # optimize constraint loss






                input, target = prefetcher.next()
                if i%args.print_freq == 0:
                    if args.local_rank == 0:
                        mean = []
                        var = []
                        for m in model.modules():
                            if isinstance(m, norm_layer):
                                mean_, var_ = m.get_mean_var()
                                mean.append(mean_.abs())
                                var.append(var_.abs())
                        mean = torch.mean(torch.stack(mean))
                        var = torch.mean(torch.stack(var))
                        curr_idx = epoch * len(train_loader) + i

                        # get the constraint weight
                        lambda_ = []
                        xi_ = []
                        for m in model.modules():
                            if isinstance(m, Constraint_Lagrangian):
                                lambda_.append(m.lambda_.data.abs().mean())
                                xi_.append(m.xi_.data.abs().mean())
                        lambda_ = torch.max(torch.stack(lambda_))
                        xi_ = torch.max(torch.stack(xi_))


                    # Every print_freq iterations, check the loss, accuracy, and speed.
                    # For best performance, it doesn't make sense to print these metrics every
                    # iteration, since they incur an allreduce and some host<->device syncs.

                    # Measure accuracy

                    # Average loss and accuracy across processes for logging

                    # to_python_float incurs a host<->device sync

                    torch.cuda.synchronize()
                    batch_time.update((time.time() - end)/args.print_freq)
                    end = time.time()
                    remain_iter = args.epochs * len(train_loader) - (epoch*len(train_loader) + i)
                    remain_time = remain_iter * batch_time.avg
                    t_m, t_s = divmod(remain_time, 60)
                    t_h, t_m = divmod(t_m, 60)
                    remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

                    if args.local_rank == 0:
                        logger.info('Epoch: [{0}][{1}/{2}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Speed {3:.3f} ({4:.3f})\t'
                            'Loss {loss.val:.10f} ({loss.avg:.4f})\t'
                            'Constraint mean {corat_mean:.4f}\t'
                            'Constraint var {corat_var:.4f}\t'
                            'Constraint lambda {corat_lambda:.4f}\t'
                            'Constraint xi {corat_xi:.4f}\t'
                            'mean {mean:.4f}\t'
                            'var {var:.4f}\t'
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                            epoch, i, len(train_loader),
                            args.world_size*args.batch_size/batch_time.val,
                            args.world_size*args.batch_size/batch_time.avg,
                            batch_time=batch_time,
                            corat_mean = -1 * weight_mean.item(),
                            corat_var = -1 * weight_var.item(),
                            corat_lambda = lambda_,
                            corat_xi = xi_,
                            mean = mean,
                            var = var,
                            loss=losses, top1=top1, top5=top5))
                        logger.info("remain time:  {}".format(remain_time))


            if idx == 0:
                track_layer = 0
                for m in model.modules():
                    if isinstance(m, norm_layer):
                        if track_layer == layer:
                            m._initialize_mu()
                            break
                        else:
                            track_layer +=1
            elif idx == 1:
                track_layer = 0
                for m in model.modules():
                    if isinstance(m, norm_layer):
                        if track_layer == layer:
                            m._initialize_gamma()
                            break
                        else:
                            track_layer += 1

            if args.world_size > 1:
                allreduce_params(model.parameters())
            for m in model.modules():
                if isinstance(m, norm_layer):
                    m.reset_norm_statistics()



def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    train_loss_avg = 0
    train_loss = AverageMeter()
    correct = 0
    total = 0
    mean = 0
    var = 0
    lambda_ = 0
    xi_ = 0


    # switch to train mode
    model.train()
    end = time.time()

    prefetcher = data_prefetcher(train_loader)
    input, target = prefetcher.next()
    i = 0
    num_channel = 0
    for m in model.modules():
        if isinstance(m, norm_layer):
            num_channel += m.num_features

    while input is not None:
        i += 1


        adjust_learning_rate(optimizer, epoch, i, len(train_loader))
        lag_weight = adjust_constraint_weight(args, epoch * len(train_loader) + i, args.epochs * len(train_loader))

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        # constraint loss
        weight_mean = 0
        weight_var = 0
        num_neg_mean = 0
        num_neg_var = 0
        num_neg_var_large = 0
        for m in model.modules():
            if isinstance(m, Constraint_Lagrangian):
                if args.constraint_weighted_average == True:
                    weight_mean_, weight_var_ =  m.get_weight_mean_var()
                else:
                    weight_mean_, weight_var_ = m.get_weight_mean_var_sum()
                weight_mean += weight_mean_
                weight_var += weight_var_
        if args.constraint_weighted_average == False:
            weight_mean /= num_channel
            weight_var /= num_channel

        constraint_loss = args.lambda_weight_mean * weight_mean + weight_var
        constraint_loss = lag_weight * constraint_loss

        # optimize constraint loss

        train_loss.update(loss.item())
        train_loss_avg += loss.item()
        if args.distributed:
                reduced_loss = reduce_tensor(loss.data)
        else:
            reduced_loss = loss.data

        # to_python_float incurs a host<->device sync
        losses.update(reduced_loss.item(), input.size(0))

        loss += constraint_loss


        if args.mixed_precision:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        p_grad = 0
        p_lag_grad = 0
        p_mu_grad = 0
        p_gamma_grad = 0
        p_affine_grad = 0
        p_theta_grad = 0
        param = ['lagrangian', 'mu_', 'gamma_', 'post_affine_layer']
        max_grads = [p_lag_grad, p_mu_grad, p_gamma_grad, p_affine_grad]
        max_param_name = None

        for p_name, p in model.named_parameters():
            max_grad = p.grad.abs().max()
            is_theta = True
            for idx, (pg_name, pg_grad) in enumerate(zip(param, max_grads)):
                if pg_name in p_name:
                    is_theta = False
                    if max_grad > max_grads[idx]:
                        max_grads[idx] = max_grad
            if is_theta:
                if max_grad > p_theta_grad:
                    p_theta_grad = max_grad
            if max_grad > p_grad:
                p_grad = max_grad
                max_param_name = p_name


        if args.local_rank == 0 and i % args.print_freq == 0:
            to_log_str = ''
            for pg_name, pg_grad in zip(param, max_grads):
                to_log_str += "{}: {:.4f}\t".format(pg_name, pg_grad)
            to_log_str += "theta: {:.4f}\t".format(p_theta_grad)
            logger.info("param max grad: {:.4f} name: {} {}".format(p_grad, max_param_name, to_log_str))
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        # for param in model.parameters():
        #     logger.info(param.data.double().sum().item(), param.grad.data.double().sum().item())

        optimizer.step()

        if i%args.print_freq == 0:
            mean = []
            var = []
            for m in model.modules():
                if isinstance(m, norm_layer):
                    mean_, var_ = m.get_mean_var()
                    mean.append(mean_.abs())
                    var.append(var_.abs())
            mean = torch.mean(torch.stack(mean))
            var = torch.mean(torch.stack(var))
            curr_idx = epoch * len(train_loader)+i
            for m in model.modules():
                if isinstance(m, norm_layer):
                    m.reset_norm_statistics()


            # get the constraint weight
            lambda_ = []
            xi_ = []
            for m in model.modules():
                if isinstance(m, Constraint_Lagrangian):
                    lambda_.append(m.lambda_.data.abs().mean())
                    xi_.append(m.xi_.data.abs().mean())
            lambda_ = torch.mean(torch.stack(lambda_))
            xi_ = torch.mean(torch.stack(xi_))



            # Every print_freq iterations, check the loss, accuracy, and speed.
            # For best performance, it doesn't make sense to print these metrics every
            # iteration, since they incur an allreduce and some host<->device syncs.

            # Measure accuracy
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            # Average loss and accuracy across processes for logging
            if args.distributed:
                prec1 = reduce_tensor(prec1)
                prec5 = reduce_tensor(prec5)

            # to_python_float incurs a host<->device sync
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            torch.cuda.synchronize()
            batch_time.update((time.time() - end)/args.print_freq)
            end = time.time()
            remain_iter = args.epochs * len(train_loader) - (epoch*len(train_loader) + i)
            remain_time = remain_iter * batch_time.avg
            t_m, t_s = divmod(remain_time, 60)
            t_h, t_m = divmod(t_m, 60)
            remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

            if args.local_rank == 0:
                logger.info("lag_weight: {}".format(lag_weight))

                logger.info("lambda: {} xi: {}".format(lambda_, xi_))
                logger.info('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Speed {3:.3f} ({4:.3f})\t'
                      'Loss {loss.val:.10f} ({loss.avg:.4f})\t'
                      'Constraint mean {corat_mean:.4f}\t'
                      'Constraint var {corat_var:.4f}\t'
                      'mean {mean:.4f}\t'
                      'var {var:.4f}\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i, len(train_loader),
                       args.world_size*args.batch_size/batch_time.val,
                       args.world_size*args.batch_size/batch_time.avg,
                       batch_time=batch_time,
                       corat_mean =  weight_mean.item(),
                       corat_var = weight_var.item(),
                       mean = mean,
                       var = var,
                       loss=losses, top1=top1, top5=top5))
                logger.info("remain time:  {}".format(remain_time))
                lrs = []
                for pg in optimizer.param_groups:
                    lrs.append(pg['lr'])
                logger.info("learning rate: {}".format(lrs))

        input, target = prefetcher.next()
    if args.local_rank == 0:
        wandb.log({"train/acc_epoch": top1.avg}, step=epoch)
        wandb.log({"train/loss_epoch": losses.avg}, step=epoch)
        wandb.log({"train/acc5_epoch": top5.avg}, step=epoch)
        wandb.log({"train/norm_mean(abs)": mean.item()}, step=epoch)
        wandb.log({"train/norm_var-1(abs)": var.item()}, step=epoch)
        wandb.log({"train/constraint_loss_mean":  weight_mean.item()}, step=epoch)
        wandb.log({"train/constraint_loss_var": weight_var.item()},step=epoch)
    torch.cuda.empty_cache()

        # Pop range "Body of iteration {}".format(i)



def validate(epoch, val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    prefetcher = data_prefetcher(val_loader)
    input, target = prefetcher.next()
    i = 0
    while input is not None:
        i += 1

        # compute output
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

        losses.update(reduced_loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # TODO:  Change timings to mirror train().
        if args.local_rank == 0 and i % args.print_freq == 0:
            logger.info('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {2:.3f} ({3:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader),
                   args.world_size * args.batch_size / batch_time.val,
                   args.world_size * args.batch_size / batch_time.avg,
                   batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

        input, target = prefetcher.next()
    if args.local_rank == 0:
        wandb.log({"test/acc_epoch": top1.avg}, step=epoch)
        wandb.log({"test/loss_epoch": losses.avg}, step=epoch)

    logger.info(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, step, len_epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 40

    if epoch >= 100:
        factor = factor + 1

    lr = args.lr*(0.1**factor)
    constraint_lr = args.constraint_lr * (0.1 ** factor)


    """Warmup"""
    if epoch < 5:
        lr = lr*float(1 + step + epoch*len_epoch)/(5.*len_epoch)
        constraint_lr = constraint_lr*float(1 + step + epoch*len_epoch)/(5.*len_epoch)



    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = constraint_lr



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt

if __name__ == '__main__':
    main()
