import os
import random

import numpy as np
import pandas as pd
import cv2
import SimpleITK as sitk
import radiomics
from tqdm import tqdm



def texture_features(image, mask, extractor, all_channels: bool=True, channel: int=0):
    """ Function to extract image texture feaures
    
        Params:
        -- image: input image
        -- mask: binary mask
        -- extractor: extractor object
        -- all_channels (bool): wheter to apply the feature extraction to all image channels (e.g., RGB) or only one channel (e.g., R)
        -- channel (int): if "all_channels" set to True, then specify which channel you want to process (Default: 0, thus 1° channel)
        
        Return:
        -- features (dict): extracted features

    """

    if all_channels==True:
        for image_channel in range(image.shape[2]):
            image_1channel = image[:,:,image_channel]
            image_1channel_stk = sitk.GetImageFromArray(image_1channel)

            features_1channel = extractor.execute(image_1channel_stk, mask, label=255)

            if image_channel == 0:
                features = {feature.split("original_")[-1]: [] for feature in features_1channel.keys() if feature.startswith("original")}

            for feature in features.keys():
                features[feature].append(features_1channel["original_"+feature])

        for feature in features.keys():
            features[feature] = np.mean(features[feature])

    else:
        image_1channel = image[:,:,channel]
        image_1channel_stk = sitk.GetImageFromArray(image_1channel)

        features_1channel = extractor.execute(image_1channel_stk, mask, label=255)

        features = {feature.split("original_")[-1]: float(features_1channel[feature]) for feature in features_1channel.keys() if feature.startswith("original")}
    
    return features



def test():
    
    params = "texture_analysis/dermaTextureAnalysis.yaml"  # path where of yaml settings file
    extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(params)      # Instantiate the extractor

    path_image_gt = f"images_lr"
    path_mask_gt = f"masks_gt"

    path_image_sr = f"images_sr_from_lr"  
    path_mask_sr = f"masks_sr"

    images_filenames_gt = os.listdir(path_image_gt)
    masks_filenames_gt = os.listdir(path_mask_gt)

    images_filenames_sr = os.listdir(path_image_sr)
    masks_filenames_sr = os.listdir(path_mask_sr)


    df_gt = pd.DataFrame()
    df_sr = pd.DataFrame()
    # df_gt.index = images_filenames_gt

    for image_filename_gt, mask_filename_gt, image_filename_sr, mask_filename_sr in tqdm(zip(images_filenames_gt, masks_filenames_gt, images_filenames_sr, masks_filenames_sr)):
        ### Set the path ###
        image_path_gt = os.path.join(path_image_gt, image_filename_gt)
        image_path_sr = os.path.join(path_image_sr, image_filename_sr)
        mask_path_gt = os.path.join(path_mask_gt, mask_filename_gt)
        mask_path_sr = os.path.join(path_mask_sr, mask_filename_sr)
        
        ### Read images and masks ###
        image_gt = cv2.imread(image_path_gt)
        image_gt = cv2.cvtColor(image_gt, cv2.COLOR_BGR2YCrCb)
        image_sr = cv2.imread(image_path_sr)
        image_sr = cv2.cvtColor(image_sr, cv2.COLOR_BGR2YCrCb)

        # Read a mask:
        # mask_gt = sitk.ReadImage(mask_path_gt, sitk.sitkUInt32)  # imageIO="PNGImageIO"
        # Set a mask equal to the image:
        mask_gt_array = np.ones((image_gt.shape[0], image_gt.shape[1]), np.uint32)*255
        mask_gt_array[0:1,0:1] = 0
        mask_gt = sitk.GetImageFromArray(mask_gt_array)

        # mask_sr = sitk.ReadImage(mask_path_sr, sitk.sitkUInt32)
        mask_sr_array = np.ones((image_sr.shape[0], image_sr.shape[1]), np.uint32)*255
        mask_sr_array[0:1,0:1] = 0
        mask_sr = sitk.GetImageFromArray(mask_sr_array)

        ### Calculate the features ###
        features_gt = texture_features(image_gt, mask_gt, extractor, all_channels=False)
        features_sr = texture_features(image_sr, mask_sr, extractor, all_channels=False)

        ### Save info using pandas Dataframe ###
        tmp_gt = pd.DataFrame([features_gt])
        df_gt = df_gt.append(tmp_gt, ignore_index = True)
        tmp_sr = pd.DataFrame([features_sr])
        df_sr = df_sr.append(tmp_sr, ignore_index = True)
        
        ### uncomment to calculate distance between features ###
        # features_gt_list = [features_gt[key] for key in features_gt.keys()]
        # features_sr_list = [features_sr[key] for key in features_sr.keys()]
        # features_dist = list(np.abs(np.array(features_gt_list) - np.array(features_sr_list)))
        # t = range(0, len(features_dist))
    

    df_gt.index = [filename.split('.')[0] for filename in images_filenames_gt]
    df_sr.index = [filename.split('.')[0] for filename in images_filenames_sr]
    print(df_gt.head())
    print(df_sr.head())

    ### Save features info as Excel file ###
    excel_path_gt = f"texture_analysis/gt.xlsx"  # path where to save gt results
    df_gt.to_excel(excel_path_gt)
    excel_path_sr = f"sr_from_lr.xlsx"  # path where to save sr results
    df_sr.to_excel(excel_path_sr)



if __name__ == "__main__":
    test()
