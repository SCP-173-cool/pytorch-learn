#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 00:08:04 2021

@author: loktarxiao
"""

import sys
sys.dont_write_bytecode = True
import os
import torch
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def get(self):
        return self.avg

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} (avg: {avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class MetricsMonitor(object):
    """
    """
    def __init__(self, filepath, net, mode="max", verbose=1, save_weights_only=False):
        """
        """
        self.net = net
        self.verbose = verbose
        self.filepath = filepath
        self.save_weights_only = save_weights_only
        self.epochs_since_last_save = 0

        os.makedirs(self.filepath, exist_ok=True)

        if mode not in ["min", "max"]:
            raise Exception

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf

    def __call__(self, metric_value, epoch):
        """
        """
        self.epochs_since_last_save += 1
        current = metric_value

        filename = os.path.join(self.filepath, "model-{:0>4d}-{:.4f}".format(epoch, metric_value))

        if self.monitor_op(current, self.best):
            print('\nEpoch %05d: metric improved from %0.5f to %0.5f,'
                  ' saving model to %s' % (epoch, self.best, current, filename))
            self.best = current

            if self.save_weights_only:
                torch.save(self.net.state_dict(), filename)
            else:
                torch.save(self.net, filename)
        else:
            print('\nEpoch %05d: metric did not improve from %0.5f' %
                                  (epoch, self.best))