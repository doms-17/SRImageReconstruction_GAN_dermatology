import os
import cv2
import pathlib
import albumentations as A
from tqdm import tqdm
from matplotlib import pyplot as plt

HIGH_RES = 1024
LOW_RES = HIGH_RES // 4
LOW_SCALE = 4


class Augmentation:
    def __init__(self, image):
        self.image = image

    def augment(self):
        transform = A.Compose([
            A.Affine(scale=(1.05), keep_ratio=True,
                     shear=[-5, 5], interpolation=cv2.INTER_CUBIC, mode=cv2.BORDER_CONSTANT, p=0.5),
            A.Flip(p=1),
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.2, p=1),
                A.RandomGamma(gamma_limit=(60, 140), p=1),
            ], p=1),
            A.OneOf([
                A.ElasticTransform(alpha=500, sigma=50, alpha_affine=10,
                                   interpolation=cv2.INTER_CUBIC, border_mode=cv2.BORDER_CONSTANT, p=1),
                A.GridDistortion(num_steps=20, distort_limit=0.05,
                                 interpolation=cv2.INTER_CUBIC, border_mode=cv2.BORDER_CONSTANT, p=1),
                A.NoOp(p=1),
            ], p=1),
            A.Compose([
                A.CenterCrop(height=HIGH_RES-50, width=HIGH_RES-50, p=1),
                A.Resize(height=HIGH_RES, width=HIGH_RES,
                         interpolation=cv2.INTER_CUBIC, p=1),
            ], p=1),
        ], p=1)
        return transform(image=self.image)['image']


class Paired:
    def __init__(self, image):
        self.image = image

    def lowres_1order(self, scale):
        lowres_1st = A.Compose([
            # -----Blur
            A.OneOf([
                A.GaussianBlur(blur_limit=(7, 21),
                               sigma_limit=(0.2, 3), p=0.7),
                A.AdvancedBlur(blur_limit=(7, 21), sigmaX_limit=(
                    0.2, 3), sigmaY_limit=(0.2, 3), beta_limit=(0.5, 4), p=0.15),
                A.RingingOvershoot(blur_limit=(7, 21), p=0.1),
            ], p=1),
            # -----Downscale
            A.OneOf([
                # A.Downscale(scale_min=1/LOW_SCALE, scale_max=1/LOW_SCALE, interpolation=cv2.INTER_NEAREST, p=1),
                A.Downscale(scale_min=1/scale, scale_max=1/scale,
                            interpolation=cv2.INTER_LINEAR, p=1),
                A.Downscale(scale_min=1/scale, scale_max=1 / \
                            scale, interpolation=cv2.INTER_AREA, p=1),
                A.Downscale(scale_min=1/scale, scale_max=1/scale,
                            interpolation=cv2.INTER_CUBIC, p=1),
            ], p=1),
            # -----Noise
            A.OneOf([
                A.GaussNoise(var_limit=(1, 30), p=0.5),
                A.GaussNoise(var_limit=(1, 30), per_channel=False, p=0.4),
                A.ISONoise(intensity=(0.05, 0.5), p=0.5),
            ], p=1),
            # -----Compression
            A.ImageCompression(quality_lower=30, quality_upper=95, p=1),
        ], p=1)
        return lowres_1st(image=self.image)['image']

    def lowres_norder(self, scale):
        lowres_1st_2nd = A.Compose([
            # -----Blur
            A.OneOf([
                A.GaussianBlur(blur_limit=(7, 21),
                               sigma_limit=(0.2, 3), p=0.7),
                A.AdvancedBlur(blur_limit=(7, 21), sigmaX_limit=(
                    0.2, 3), sigmaY_limit=(0.2, 3), beta_limit=(0.5, 4), p=0.15),
                A.RingingOvershoot(blur_limit=(7, 21), p=0.1),
            ], p=0.8),
            # -----Downscale
            A.OneOf([
                # A.Downscale(scale_min=1/scale, scale_max=1/scale, interpolation=cv2.INTER_NEAREST, p=1),
                A.Downscale(scale_min=1/scale, scale_max=1/scale,
                            interpolation=cv2.INTER_LINEAR, p=1),
                A.Downscale(scale_min=1/scale, scale_max=1/scale,
                            interpolation=cv2.INTER_AREA, p=1),
                A.Downscale(scale_min=1/scale, scale_max=1/scale,
                            interpolation=cv2.INTER_CUBIC, p=1),
            ], p=1),
            # -----Noise
            A.OneOf([
                A.GaussNoise(var_limit=(1, 25), p=0.5),
                A.GaussNoise(var_limit=(1, 30), per_channel=False, p=0.4),
                A.ISONoise(intensity=(0.05, 0.1), p=0.5),
            ], p=1),
            # -----Compression
            A.OneOf([
                A.Compose([
                    A.ImageCompression(
                        quality_lower=30, quality_upper=95, p=1),
                    A.RingingOvershoot(blur_limit=(7, 21), p=0.8),
                ], p=1),
                A.Compose([
                    A.RingingOvershoot(blur_limit=(7, 21), p=0.8),
                    A.ImageCompression(
                        quality_lower=30, quality_upper=95, p=1),
                ], p=1),
            ], p=1),
        ], p=1)
        return lowres_1st_2nd(image=self.image)['image']
