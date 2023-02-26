import os
import random

import cv2
from dermasr.paired import Paired
from tqdm import tqdm



def test():
    ### Set path ###
    root_folder_path: str = "D:/DOMI/University/Thesis/Coding/Dataset/TestSet/"
    test_set: str = "Novara_good"
    new_image_folder: str = f"{test_set}_paired"
    os.makedirs(os.path.join(root_folder_path,new_image_folder), exist_ok=True)

    ### Get all images ###
    image_filenames = os.listdir(os.path.join(root_folder_path,test_set))

    for image_filename in tqdm(image_filenames):
        ### Read image ###
        image_gt = cv2.imread(os.path.join(root_folder_path,test_set,image_filename))

        paired = Paired(image_gt) # instantiate the paired object
        ### Degradate image ###
        image_lr = paired.degradation_first_order(scale=4)    # 1st order degradation

        random_scale: int = random.randint(2,4)
        image_lr = Paired(image_lr).degradation_n_order(scale=random_scale)     # 2nd order degradation
        # random_scale: int = random.randint(2,4)
        # image_lr = Paired(image_lr).degradation_n_order(scale=random_scale)     # 3rd order degradation etc.

        ### Enhance image ###
        image_hr = paired.enhance()

        ### Save image ###
        cv2.imwrite(os.path.join(root_folder_path,new_image_folder,image_filename), image_lr)
        cv2.imwrite(os.path.join(root_folder_path,new_image_folder,image_filename), image_hr)



if __name__ == "__main__":
    test()







