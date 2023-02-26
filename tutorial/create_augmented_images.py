import os
import random

import cv2
from dermasr.augmentation import Augmentation
from tqdm import tqdm



def test():
    ### Set path ###
    root_folder_path: str = "D:/DOMI/University/Thesis/Coding/Dataset/TestSet/"
    test_set: str = "Novara_good"
    new_image_folder: str = f"{test_set}_augemnted"
    os.makedirs(os.path.join(root_folder_path,new_image_folder), exist_ok=True)

    ### Get all images ###
    image_filenames = os.listdir(os.path.join(root_folder_path,test_set))

    for image_filename in tqdm(image_filenames):
        ### Read image ###
        image_gt = cv2.imread(os.path.join(root_folder_path,test_set,image_filename))
        augmentation = Augmentation(image_gt)

        ### Augment image ###
        image_augmented = augmentation.augment(prob_centralCropResize=0)

        ### Save image ###
        cv2.imwrite(os.path.join(root_folder_path,new_image_folder,image_filename), image_augmented)



if __name__ == "__main__":
    test()







