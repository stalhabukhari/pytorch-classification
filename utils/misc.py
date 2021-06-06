'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import errno
import os
import sys
import time
import math, csv

import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

__all__ = ['get_mean_and_std', 'init_params', 'mkdir_p', 'AverageMeter', 'CSVLogger']


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
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


class CSVLogger(object):
    def __init__(self, path, fields=None):
        if fields is None:
            fields = ['Arch', 'Dataset', 'CheckpointPath', 'LoadModel',
                    'Loss', 'Top1Acc', 'Top5Acc']
        self.fields = fields
        if not os.path.isfile(path):
            self.write_to_csv(None, path, write_header=True)
        self.path = path

    def __call__(self, data_row):
        self.check_keys(data_row, self.fields)
        self.write_to_csv(data_row, self.path)

    def write_to_csv(self, data_row, file_name, write_header=False):
        write_mode = 'w' if write_header else 'a'
        with open(file_name, mode=write_mode, newline='') as file:
            file_writer = csv.DictWriter(file, fieldnames=self.fields)
            if write_header:
                file_writer.writeheader()
            else:
                file_writer.writerow(data_row)

    @staticmethod
    def check_keys(dc_in, l_ref):
        """"""
        lin, lref = list(dc_in.keys()), list(l_ref)
        lin.sort()
        lref.sort()
        assert lin == lref
