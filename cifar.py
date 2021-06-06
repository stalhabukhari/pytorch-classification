'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Mode (train/test)
parser.add_argument("--train", action="store_true")
parser.add_argument("--test", action="store_true")
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--loadModel', type=str, default='model_best.pth.tar')
# parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
#                     help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Mode (train/test)
assert args.train or args.test
assert not (args.train and args.test)

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
    torch.backends.cudnn.benchmark = True

best_acc = 0  # best test accuracy
abs_step_count = 0 # count training iterations

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)



    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100

    if args.train:
        trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
        train_idx, valid_idx = train_test_split(
            np.arange(len(trainset.targets)),
            test_size=0.1,
            shuffle=True,
            stratify=trainset.targets,
            random_state=args.manualSeed)

        # valid_size = 0.1 # 10%
        # indices = list(range(len(trainset)))
        # split = int(np.floor(valid_size * len(trainset)))
        # np.random.seed(args.manualSeed)
        # np.random.shuffle(indices)
        # train_idx, valid_idx = indices[split:], indices[:split]
        trainset_train = data.Subset(trainset, train_idx) # training set
        trainset_valid = data.Subset(trainset, valid_idx) # validation set

        trainloader_all = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
        trainloader = data.DataLoader(trainset_train, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
        validloader = data.DataLoader(trainset_valid, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
        print(f' Data size: {len(trainset)} | Train Set: {len(trainset_train)} | Valid Set: {len(trainset_valid)} ')

    if args.test:
        testset = dataloader(root='./data', train=False, download=True, transform=transform_test)
        testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
        print(f' Test Set size: {len(testset)}')

    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    cardinality=args.cardinality,
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('densenet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    growthRate=args.growthRate,
                    compressionRate=args.compressionRate,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('wrn'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    block_name=args.block_name,
                )
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)

    model = torch.nn.DataParallel(model).cuda()
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nn.CrossEntropyLoss()

    if args.test:
        loaded_ckpt = load_checkpoint(checkpoint=args.checkpoint, filename=args.loadModel)
        model.load_state_dict(loaded_ckpt['model_state'])

        test_loss, test_acc = test(testloader, model, criterion, None, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))

    if args.train:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        # tensorboard writer:
        tbwriter = SummaryWriter(log_dir=args.checkpoint) # read from config
        # log graph to tensorboard:
        with torch.no_grad():
            tbwriter.add_graph(model, torch.zeros(5, 3, 32, 32).cuda())
        print('Created TensorBoard instance')

        # Resume
        title = 'cifar-10-' + args.arch
        if args.resume:
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
            args.checkpoint = os.path.dirname(args.resume)
            checkpoint = torch.load(args.resume)
            best_acc = checkpoint['best_acc']
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
        else:
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
            logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])


        # Train and Validate
        for epoch in range(start_epoch, args.epochs):
            adjust_learning_rate(optimizer, epoch)

            print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

            if epoch+1==args.epochs: # final epoch
                print('>> Training over the complete training dataset')
                train_loss, train_acc = train(trainloader_all, model, criterion, optimizer, epoch, use_cuda) # ,tbwriter)
            else:
                train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda) # ,tbwriter)
            test_loss, test_acc = test(validloader, model, criterion, epoch, use_cuda)

            # append logger file
            logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

            hist_dict = {
                'loss/train': train_loss,
                'loss/valid': test_loss,
                'acc/train': train_acc,
                'acc/valid': test_acc,
            }
            lr_dict = dict((f'lr_epoch/group_{g}', param_grp['lr']) \
                        for g, param_grp in enumerate(optimizer.param_groups))
            hist_dict.update(lr_dict)

            log_to_tensorboard(tbwriter, hist_dict, epoch)

            # save model
            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)
            save_checkpoint({
                    'epoch': epoch + 1,
                    'model_state': model.state_dict(),
                    'acc': test_acc,
                    'best_acc': best_acc,
                    'opt_state' : optimizer.state_dict(),
                }, is_best, checkpoint=args.checkpoint)

        logger.close()
        logger.plot()
        savefig(os.path.join(args.checkpoint, 'log.eps'))

        print('Best acc:')
        print(best_acc)


def train(trainloader, model, criterion, optimizer, epoch, use_cuda, tbwriter=None):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
        #inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        if tbwriter:
            lr_dict = dict((f'lr_epoch/group_{g}', param_grp['lr']) \
                        for g, param_grp in enumerate(optimizer.param_groups))
            log_to_tensorboard(tbwriter, lr_dict, abs_step_count)
            abs_step_count += 1

        # compute output
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            loss = sum(criterion(out_i, targets) for out_i in outputs)
        else:
            loss = criterion(outputs, targets)

        # measure accuracy and record loss
        if isinstance(outputs, tuple): # average outputs (ensemble)
            outputs = sum(F.softmax(out_i, dim=1) for out_i in outputs)/float(len(outputs))
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def test(testloader, model, criterion, epoch, use_cuda, tbwriter=None):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        # compute output
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            loss = sum(criterion(out_i, targets) for out_i in outputs)
        else:
            loss = criterion(outputs, targets)

        # measure accuracy and record loss
        if isinstance(outputs, tuple): # average outputs (ensemble)
            outputs = sum(F.softmax(out_i, dim=1) for out_i in outputs)/float(len(outputs))
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def load_checkpoint(checkpoint='checkpoint', filename='model_best.pth.tar'):
    assert '.pth.tar' in filename
    filepath = os.path.join(checkpoint, filename)
    return torch.load(filepath)

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

def log_to_tensorboard(writer, scalar_dict, step):
    """routine for tensorboard logging"""
    for key in scalar_dict:
        writer.add_scalar(key, scalar_dict[key], step)


if __name__ == '__main__':
    main()
