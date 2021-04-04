#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 00:08:04 2021

@author: loktarxiao
"""

from process_functions import *
from augment import strong_aug
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import random
import os
import sys
sys.dont_write_bytecode = True


def get_file_lst(mode="train", fold=1, shuffle=True):
    """ Get file list
    """
    root = "/apdcephfs/private_loktarxiao/datasets/dogs_vs_cats"
    df = pd.read_csv(os.path.join(root, "split_df.csv"))

    # Filter Dataframe via fold and mode
    if mode == "train":
        df = df[df["fold"] != fold]
    elif mode == "valid":
        df = df[df["fold"] == fold]

    # Read image path
    img_path_lst = [os.path.join(root, i) for i in df.img_path.values]
    label_lst = list(df.class_id.values)
    item_lst = list(zip(img_path_lst, label_lst))

    # Shuffle item list
    if shuffle:
        random.shuffle(item_lst)

    print("Got {:d} items in {} mode.".format(len(item_lst), mode))
    return item_lst


class BaseImageClsDataset(Dataset):
    """
    """

    def __init__(self,
                 item_lst,
                 img_transform=None,
                 trg_transform=None):
        self.item_lst = item_lst
        self.img_transform = img_transform
        self.trg_transform = trg_transform

    def __getitem__(self, idx):
        """
        """
        img_path, target = self.item_lst[idx]

        img = image_read(img_path)
        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.trg_transform is not None:
            target = self.trg_transform(target)

        return img, target

    def __len__(self):
        return len(self.item_lst)


aug = strong_aug(p=0.6)
def train_img_func(img):
    """ train image transform function
    """
    img = resize_longer_edge(img, 256)
    img = square_padding(img, mode="noise", target_size=256)
    img = aug(image=img)["image"]
    img = random_crop(img, crop_shape=(224, 224))
    img = random_mask(img, num_mask=20, min_size=5, max_size=50)

    img = normalize(img, 0, 255)
    return img.transpose([2, 0, 1])


def valid_img_func(img):
    """validation image transform function
    """
    img = resize_longer_edge(img, 256)
    img = square_padding(img, mode="black", target_size=256)
    img = cv2.resize(img, (224, 224))

    img = normalize(img, 0, 255)
    return img.transpose([2, 0, 1])


def get_dataloader(config):
    bs = config["batch_size"]
    workers = config["workers"]

    train_lst = get_file_lst(mode="train", fold=0, shuffle=True)
    valid_lst = get_file_lst(mode="valid", fold=0, shuffle=False)

    train_dataset = BaseImageClsDataset(
        train_lst, img_transform=train_img_func)
    valid_dataset = BaseImageClsDataset(
        valid_lst, img_transform=valid_img_func)

    train_loader = DataLoader(train_dataset,
                              batch_size=bs,
                              shuffle=True,
                              num_workers=workers,
                              pin_memory=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=bs,
                              shuffle=False,
                              num_workers=workers,
                              pin_memory=True)

    return train_lst, valid_lst, train_dataset, valid_dataset, train_loader, valid_loader
