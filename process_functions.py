#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 06:08:10 2019

@author: loktarxiao
"""
import sys
sys.dont_write_bytecode = True

import numpy as np
import cv2
import random
from skimage.draw import random_shapes

from sklearn.utils import shuffle

def image_read(img_path, mode="RGB"):
    """ The faster image reader with opencv API.
    """
    mode = mode.lower()
    with open(img_path, 'rb') as fp:
        raw = fp.read()
        if mode == 'rgb':
            img = cv2.imdecode(np.asarray(bytearray(raw), dtype="uint8"), cv2.IMREAD_COLOR)
            img = img[:,:,::-1]
        elif mode == 'gray' or mode == 'grey':
            img = cv2.imdecode(np.asarray(bytearray(raw), dtype="uint8"), cv2.IMREAD_GRAYSCALE)
    return img

def normalize(img, a, b):
    """ Normalizations
    Return (img - a) / b
    """
    img = img.astype(np.float32)
    return (img - a) / b

def convert_onehot(sparse_label, length):
    """
    """
    label = np.zeros(length)
    label[sparse_label] = 1.
    return label

def resize_shorter_edge(img, length):
    """ Resize image to target shorter_edge.
    """
    rows, cols = img.shape[:2]
    if rows >= cols:
        size = (int(length), int(1.0 * rows * length / cols))
    else:
        size = (int(1.0 * cols * length / rows), int(length))
    img = cv2.resize(img, size,
                     interpolation=cv2.INTER_CUBIC)
    return img

def resize_longer_edge(img, length):
    """ Resize image to target longer_edge.
    """
    rows, cols = img.shape[:2]
    if rows <= cols:
        size = (int(length), int(1.0 * rows * length / cols))
    else:
        size = (int(1.0 * cols * length / rows), int(length))
    img = cv2.resize(img, size,
                     interpolation=cv2.INTER_CUBIC)
    return img

def random_crop(image, mask=None, crop_shape=(224, 224)):
    """ Crop patch from image randomly.
    """
    oshape = np.shape(image)
    nh = random.randint(0, oshape[0] - crop_shape[0])
    nw = random.randint(0, oshape[1] - crop_shape[1])
    image_crop = image[nh:nh + crop_shape[0], nw:nw + crop_shape[1]]
    if mask is not None:
        mask_crop = mask[nh:nh + crop_shape[0], nw:nw + crop_shape[1]]
        return image_crop, mask_crop
    return image_crop

def central_crop(image, mask=None, crop_shape=(224, 224)):
    """ Crop central patch from image. 
    """
    oshape = np.shape(image)
    
    nh = int((oshape[0] - crop_shape[0]) / 2)
    nw = int((oshape[1] - crop_shape[1]) / 2)
    
    image_crop = image[nh:nh + crop_shape[0], nw:nw + crop_shape[1]]
    
    if mask is not None:
        mask_crop = mask[nh:nh + crop_shape[0], nw:nw + crop_shape[1]]
        return image_crop, mask_crop
    
    return image_crop

def random_mask(image, mask=None, num_mask=20, min_size=5, max_size=128):
    """ Random Erase function
        Reference: https://arxiv.org/pdf/1708.04896.pdf
    """
    raw, _ = random_shapes(image.shape[:2],
                           num_mask,
                           min_shapes=2,
                           min_size=min_size,
                           max_size=max_size,
                           multichannel=False,
                           allow_overlap=True)
    mask_raw = 1 - raw/ 255.
    noise = np.random.random(image.shape)
    image[mask_raw > 0] = noise[mask_raw > 0] * random.randint(0, 255)
    if mask is not None:
        mask = mask * (mask_raw == 0)
        return image, mask
    
    return image

def square_padding(img, mode="black", target_size=None):
    """ Padding image to square.
    """
    h, w, _ = img.shape

    if target_size is not None:
        if mode == "black":
            background = np.zeros((target_size, target_size, 3), np.uint8)
        elif mode == "noise":
            background = np.random.random((target_size, target_size, 3)) * random.randint(0, 255)
            background = background.astype(np.uint8)
        h_edge = int((target_size - h) / 2)
        w_edge = int((target_size - w) / 2)
        background[h_edge:h_edge+h, w_edge:w_edge+w, :] = img

    else:
        if h > w:
            if mode == "black":
                background = np.zeros((h, h, 3), np.uint8)
            elif mode == "noise":
                background = np.random.random((h, h, 3)) * random.randint(0, 255)
                background = background.astype(np.uint8)
            edge = int((h - w) / 2)
            background[:, edge:edge+w, :] = img
        elif w > h:
            if mode == "black":
                background = np.zeros((w, w, 3), np.uint8)
            elif mode == "noise":
                background = np.random.random((w, w, 3)) * random.randint(0, 255)
                background = background.astype(np.uint8)
            edge = int((w - h) / 2)
            background[edge:edge+h, :, :] = img
        else:
            return img
    
    return background

def mix_up(images, labels):
    #alpha = 1.0
    alpha = random.random()
    s_images, s_labels = shuffle(images, labels)

    lam = np.random.beta(alpha, alpha)
    new_images = lam * images + (1 - lam) * s_images
    new_labels = lam * labels + (1 - lam) * s_labels
    return new_images, new_labels


def fourier_domain_adaptation(img, target_lst, beta_limit=[1e-5, 0.24]):
    """
    """
    beta = random.uniform(beta_limit[0], beta_limit[1])
    img = np.squeeze(img)
    target_img = random.choice(target_lst)
    target_img = np.squeeze(target_img)

    target_img = cv2.resize(target_img, (img.shape[1], img.shape[0]))

    # get fft of both source and target
    fft_src = np.fft.fft2(img.astype(np.float32), axes=(0, 1))
    fft_trg = np.fft.fft2(target_img.astype(np.float32), axes=(0, 1))

    # extract amplitude and phase of both fft-s
    amplitude_src, phase_src = np.abs(fft_src), np.angle(fft_src)
    amplitude_trg = np.abs(fft_trg)

    # mutate the amplitude part of source with target
    amplitude_src = np.fft.fftshift(amplitude_src, axes=(0, 1))
    amplitude_trg = np.fft.fftshift(amplitude_trg, axes=(0, 1))
    height, width = amplitude_src.shape[:2]
    border = np.floor(min(height, width) * beta).astype(int)
    center_y, center_x = np.floor([height / 2.0, width / 2.0]).astype(int)

    y1, y2 = center_y - border, center_y + border + 1
    x1, x2 = center_x - border, center_x + border + 1

    amplitude_trg[y1:y2, x1:x2] = amplitude_src[y1:y2, x1:x2]
    amplitude_src = np.fft.ifftshift(amplitude_trg, axes=(0, 1))
    
    # get mutated image
    src_image_transformed = np.fft.ifft2(amplitude_src * np.exp(1j * phase_src), axes=(0, 1))
    src_image_transformed = np.real(src_image_transformed)

    return random.choice([src_image_transformed.astype(np.uint8), normalize_aa(src_image_transformed)])

def normalize_aa(arr):
    new_arr = ((arr - arr.min()) * (1/(arr.max() - arr.min()) * 255)).astype('uint8')
    return new_arr
