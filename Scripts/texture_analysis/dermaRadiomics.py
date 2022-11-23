import os
import random
import typing

import numpy as np
import matplotlib.pyplot as plt
import skimage
import scipy.ndimage
import cv2
import six
import SimpleITK as sitk
import radiomics
from color_space_utils import rgb2ycbcr


def texture_features(image_path: str, mask_path: str, extractor, y_only: bool=False):
    image_gt = cv2.imread(image_path)
    image_gt = cv2.cvtColor(image_gt, cv2.COLOR_BGR2YCrCb)
    mask_gt = sitk.ReadImage(mask_path, sitk.sitkUInt32)  # imageIO="PNGImageIO"

    if y_only==False:
        # image_gt = rgb2ycbcr(image_gt)
        for image_channel in range(image_gt.shape[2]):
            image_gt_1channel = image_gt[:,:,image_channel]
            image_gt_1channel_stk = sitk.GetImageFromArray(image_gt_1channel)

            features_1channel = extractor.execute(image_gt_1channel_stk, mask_gt, label=255)

            if image_channel == 0:
                info: typing.Dict(typing.Union(str,typing.Any)) = {feature: [] for feature in features_1channel.keys() if not feature.startswith("original")}
                features: typing.Dict(typing.Union(str,typing.Any)) = {feature: [] for feature in features_1channel.keys() if feature.startswith("original")}

            for feature in features.keys():
                features[feature].append(features_1channel[feature])

        for feature in features.keys():
            features[feature] = np.mean(features[feature])

    else:
        image_gt_1channel = image_gt[:,:,1]
        image_gt_1channel_stk = sitk.GetImageFromArray(image_gt_1channel)

        features_1channel = extractor.execute(image_gt_1channel_stk, mask_gt, label=255)

        info: typing.Dict(typing.Union(str,typing.Any)) = {feature: features_1channel[feature] for feature in features_1channel.keys() if not feature.startswith("original")}
        features: typing.Dict(typing.Union(str,typing.Any)) = {feature: float(features_1channel[feature]) for feature in features_1channel.keys() if feature.startswith("original")}
    
    return features


def test():
    params: str = "D:/DOMI/Github/WebRepositories/pyradiomics-master/examples/exampleSettings/Params.yaml"
    extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(params)      # Instantiate the extractor

    image_dataset = "ISBI2016_ISIC_Part1_Training_Data"
    mask_dataset = "ISBI2016_ISIC_Part1_Training_GroundTruth"

    path_image_gt: str = f"D:/DOMI/University/Thesis/Coding/Dataset/Texture_analysis/gt_images"
    path_mask_gt: str = f"D:/DOMI/University/Thesis/Coding/Dataset/Texture_analysis/gt_masks"

    path_image_sr: str = f"D:/DOMI/University/Thesis/Coding/Dataset/Texture_analysis/sr_images"  
    path_mask_sr: str = f"D:/DOMI/University/Thesis/Coding/Dataset/Texture_analysis/sr_masks"

    images_filenames_gt: typing.List[str] = os.listdir(path_image_gt)
    masks_filenames_gt: typing.List[str] = os.listdir(path_mask_gt)

    images_filenames_sr: typing.List[str] = os.listdir(path_image_sr)
    masks_filenames_sr: typing.List[str] = os.listdir(path_mask_sr)

    for image_filename_gt, mask_filename_gt, image_filename_sr, mask_filename_sr  in zip(images_filenames_gt, masks_filenames_gt, images_filenames_sr, masks_filenames_sr):
        image_path_gt = os.path.join(path_image_gt, image_filename_gt)
        mask_path_gt = os.path.join(path_mask_gt, mask_filename_gt)
        features_gt = texture_features(image_path_gt, mask_path_gt, extractor, y_only=True)

        image_path_sr = os.path.join(path_image_sr, image_filename_sr)
        mask_path_sr = os.path.join(path_mask_sr, mask_filename_sr)
        features_sr = texture_features(image_path_sr, mask_path_sr, extractor)

        features_gt_list = [features_gt[key] for key in features_gt.keys()]
        features_sr_list = [features_sr[key] for key in features_sr.keys()]


        plt.figure(figsize=(20,20))
        plt.subplot(3,1,1)
        plt.plot(features_gt_list)
        # plt.yscale('log')
        # plt.xscale(features_sr_keys)
        plt.title(f"Image: {image_filename_gt}, Features GT")
                
        plt.subplot(3,1,2)
        plt.plot(features_sr_list)
        # plt.yscale('log')
        plt.title(f"Image: {image_filename_sr}, Features SR")

        plt.subplot(3,1,3)
        plt.plot(features_gt_list - features_sr_list)
        # plt.yscale("log")
        plt.title("Difference")
        plt.show()


if __name__ == "__main__":
    test()