#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 00:08:04 2021

@author: loktarxiao
"""

import sys
sys.dont_write_bytecode = True
import torch
import numpy as np

EPS = 1e-9

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res
    
    
class ConfusionMatric(object):
    """
    """
    def __init__(self, num_classes, name):
        self.num_classes = num_classes
        self.mat = None
        self.name = name
        
    def update(self, y_true, y_pred):
        y_true = y_true.numpy()
        y_pred = y_pred.numpy()
        
        n = self.num_classes
        if self.mat is None:
            self.mat = np.zeros((n, n), dtype=np.int64)
            
        k = (y_true >= 0) & (y_true < n)
        inds = n * y_true[k] + y_pred[k]
        self.mat += np.bincount(inds, minlength=n**2).reshape(n, n)
            
    def reset(self):
        n = self.num_classes
        self.mat = np.zeros((n, n), dtype=np.int64)
        
    def compute(self):
        m = self.mat
        acc_global = m.diagonal().sum() / (m.sum() + EPS)
        acc = m.diagonal() / (m.sum(1) + EPS)
        
        return m, acc_global, acc