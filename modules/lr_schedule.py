#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 23:18:19 2020

@author: loktarxiao
"""

import sys
sys.dont_write_bytecode = True

import math
class CircleWarmUpConsineDecay():
    """
    """
    def __init__(self, start_lr, min_lr, max_lr, warmup_steps, cooldown_steps, verbose=1):
        r"""
        """
        self.start_lr = start_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.wm_steps = warmup_steps
        self.cd_steps = cooldown_steps
        self.verbose = verbose
        
        self.init_wm_k = (self.max_lr - self.start_lr) / self.wm_steps
        self.wm_k = (self.max_lr - self.min_lr) / self.wm_steps
        
    def __call__(self, epoch):
        r"""
        """
        if epoch <= self.wm_steps + self.cd_steps:
            if epoch <= self.wm_steps:
                lr = self.start_lr + self.init_wm_k * epoch
            else:
                unit_cycle = (1 + math.cos((epoch - self.wm_steps) * math.pi / self.cd_steps)) / 2
                lr = (unit_cycle * (self.max_lr - self.min_lr)) + self.min_lr
        else: 
            token = epoch % (self.wm_steps + self.cd_steps)
            if token <= self.wm_steps:
                lr = self.min_lr + self.wm_k * token
            else:
                unit_cycle = (1 + math.cos((token - self.wm_steps) * math.pi / self.cd_steps)) / 2
                lr = (unit_cycle * (self.max_lr - self.min_lr)) + self.min_lr
                
        if self.verbose == 1:
            print("Setting Learning Rate to {:.7f}".format(lr))
        
        return lr