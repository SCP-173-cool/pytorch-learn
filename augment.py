#!/usr/bin/env python
# -*- coding: utf-8 -*-
# loktarxiao @ 2019-05-06 14:42:54

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, RandomSizedCrop,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, GaussNoise,
    RGBShift, RandomRain, RandomSnow, RandomShadow, RandomFog, ElasticTransform, HistogramMatching,
    FDA
)

def strong_aug(p=0.5):
    
    return Compose([
        RandomRotate90(),
        Flip(),
        Transpose(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.8),
        OneOf([
            MotionBlur(p=0.5),
            MedianBlur(blur_limit=3, p=0.5),
            Blur(blur_limit=3, p=0.5),
        ], p=0.8),
        ShiftScaleRotate(shift_limit=0.25, rotate_limit=180, p=0.9),
        OneOf([
            OpticalDistortion(p=0.5),
            GridDistortion(p=0.5),
            IAAPiecewiseAffine(p=0.5),
            ElasticTransform(p=0.5),
        ], p=0.7),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),
        ], p=0.5),
        OneOf([
            GaussNoise(),
            RandomRain(p=0.5, brightness_coefficient=0.9, drop_width=1, blur_value=5),
            RandomSnow(p=0.5, brightness_coeff=0.5, snow_point_lower=0.1, snow_point_upper=0.3),
            RandomShadow(p=0.5, num_shadows_lower=1, num_shadows_upper=1,
                        shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1)),
            RandomFog(p=0.5, fog_coef_lower=0.3, fog_coef_upper=0.5, alpha_coef=0.1)
        ], p=0.0),
        RGBShift(p=0.9),
        HueSaturationValue(hue_shift_limit=120, sat_shift_limit=60, val_shift_limit=30, p=0.6)
    ], p=p)
