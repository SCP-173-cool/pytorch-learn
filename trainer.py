#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 00:08:04 2021

@author: loktarxiao
"""

import sys
sys.dont_write_bytecode = True
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import torch
from tqdm import tqdm
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import models

from data_io import get_dataloader
from modules.metrics import accuracy
from modules.utils import AverageMeter, MetricsMonitor
from modules.metrics import ConfusionMatric

class BaseTrainer(object):
    """
    """
    def __init__(self, cfg):
        """
        """
        # Parameter Configures
        self.train_loader, self.valid_loader = cfg["loader"]
        self.batch_size = cfg["batch_size"]
        self.model = cfg["model"]
        self.device = cfg["device"]
        self.optimizer = cfg["optimizer"]
        self.criterion = cfg["criterion"]
        self.epoch_num = cfg["epoch_num"]
        self.num_classes = cfg["num_classes"]
        self.log_interval = cfg["log_interval"]
        self.monitor = cfg["monitor"]
        
    def prepare(self):
        """ Run before training"""
        self.metrics_setting()

    def metrics_setting(self):
        """ Setting training and validation metrics"""
        self.train_acc_func = ConfusionMatric(self.num_classes, "train_acc")
        self.valid_acc_func = ConfusionMatric(self.num_classes, "valid_acc")

        self.train_loss = AverageMeter("train_loss")
        self.valid_loss = AverageMeter("valid_loss")
        self.train_acc = AverageMeter("train_acc")
        self.valid_acc = AverageMeter("valid_acc")

        
        self.train_metrics_lst = [self.train_loss,
                                  self.train_acc]
        self.valid_metrics_lst = [self.valid_loss,
                                  self.valid_acc]
    
    def before_train(self):
        """ Sometime excute before training """
        self.model.train()
        self.train_acc_func.reset()
        for met in self.train_metrics_lst:
            met.reset()

    def train_epoch(self):
        """ Training in one epoch """
        self.btic = time.time()

        for self.idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.train_step(data, target) 
        
    def train_step(self, data, target):
        """ Training in one step """
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = self.criterion(output, target)
        loss.backward()
        
        self.optimizer.step()
        self.train_acc_func.update(target.cpu(), output.argmax(axis=-1).cpu())
        self.train_loss.update(loss.item())
        
        if (self.idx + 1) % self.log_interval == 0:
            m, acc_global, acc = self.train_acc_func.compute()

            speed = self.batch_size * self.log_interval / (time.time() - self.btic)
            print('Epoch[{:0>4d}]\tBatch [{:0>4d}]\tSpeed: {:0>3d} samples/sec'.format(
                   self.epoch, self.idx + 1, int(speed))+'\tloss: {:.5f}\taccuracy: {:.5f}'.format(
                       self.train_loss.get(), acc_global))
            self.btic = time.time()

    def after_train(self):
        """ Sometime excute after training """
        m, acc_global, acc = self.train_acc_func.compute()
        self.train_acc.update(acc_global)
        
    
    def before_valid(self):
        """ Sometime excute before validation """
        print()
        print("Validatation ...")
        self.model.eval()
        self.valid_acc_func.reset()
        for met in self.valid_metrics_lst:
            met.reset()

    def valid_epoch(self):
        """ valid in one epoch """
        with torch.no_grad():
            for data, target in tqdm(self.valid_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                self.valid_acc_func.update(target.cpu(), output.argmax(axis=-1).cpu())
                self.valid_loss.update(loss.item())
        
    def after_valid(self):
        """ Sometime excute after training """
        m, acc_global, acc = self.valid_acc_func.compute()
        self.valid_acc.update(acc_global)

        for met in self.train_metrics_lst:
            met_name, value = met.name, met.get()
            print('[Epoch {:0>4d}] {}: {:.4f}'.format(self.epoch, met_name, value))

        for met in self.valid_metrics_lst:
            met_name, value = met.name, met.get()
            print('[Epoch {:0>4d}] {}: {:.4f}'.format(self.epoch, met_name, value))

        self.monitor(self.valid_acc.get(), self.epoch)
        

    def run(self):

        # Prepare works in beginning.
        self.prepare()    

        for self.epoch in range(self.epoch_num):
            tic = time.time()
            print()
            print('[Epoch %d] Starting...' % self.epoch)
            # Training process
            self.before_train()
            self.train_epoch()
            self.after_train()
            
            # Validation process
            self.before_valid()
            self.valid_epoch()
            self.after_valid()

            print('[Epoch {:0>4d}] time cost: {:.2f} sec'.format(self.epoch, time.time()-tic))
        

config = {}
config["batch_size"] = 64
config["num_classes"] = 2
config["log_interval"] = 20
config["epoch_num"] = 1500
config["workers"] = 8

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, config["num_classes"])
device = torch.device("cuda:0")
model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.00001, momentum=0.99)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


train_lst, valid_lst, train_dataset, valid_dataset, train_loader, valid_loader = get_dataloader(config)
config["loader"] = (train_loader, valid_loader)
config["model"] = model_ft
config["device"] = device
config["criterion"] = criterion
config["optimizer"] = optimizer_ft
config["monitor"] = MetricsMonitor("save_path", model_ft, mode="max")

trainer = BaseTrainer(config)
trainer.run()
