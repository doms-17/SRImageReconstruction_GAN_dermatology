import os
import random

import numpy as np
import cv2
from dermasr.artifacts import Artifacts
from tqdm import tqdm


### Set some constant variable to be used in the Preprocessing pipeline ###
HIGH_RES: int = 512      # dimension of images to select and process (Put to zero to process images of every dimension)
BORDER_THICK: int = 10       # thickness to check on borders when checking for Dark Corner
DARK_THRESH: int = 10    # threshold to classify a border as near-black or not
THRESH_RATIO: float = 1000     # threshold to activate rect2square_image_splitting
COMPRESSION: int = 6     # chose the % compression within [0,9], when saving images (NB: the more, the higher quality, the higher time computation)


def test():
    ### Set path ###
    root_folder_path: str = "D:/DOMI/University/Thesis/Coding/Dataset/TestSet_original/"
    test_set: str = "Novara dermoscopio good"
    new_image_folder: str = f"{test_set}_noArtifact"
    os.makedirs(os.path.join(root_folder_path,new_image_folder), exist_ok=True)

    ### Get all images ###
    image_filenames: list[str] = os.listdir(os.path.join(root_folder_path,test_set))

    for image_filename in tqdm(image_filenames):
        ### Read image ###
        id_image = image_filename.split('.')[0]
        image_gt = cv2.imread(os.path.join(root_folder_path,test_set,image_filename))
        height, width = image_gt.shape[0], image_gt.shape[1]

        artifact = Artifacts(image_gt)
        mask = artifact.get_mask(morph_close_kernel=50)

        ### Process images with only selected dimension ###
        if (height >= HIGH_RES and width >= HIGH_RES):
            # or (height == width and 800<height<HIGH_RES and 800<width<HIGH_RES) \
            # or (height == HIGH_RES and 700<width<HIGH_RES) or (width == HIGH_RES and 700<height<HIGH_RES):

            try:
                ### Check for Dark Corner Artifact ###
                # Calculate the mean intensity of each border of the mask
                sx_border = np.mean(mask[:, 0:BORDER_THICK])
                dx_border = np.mean(mask[:, -BORDER_THICK:])
                up_border = np.mean(mask[0:BORDER_THICK, :])
                down_border = np.mean(mask[-BORDER_THICK:, :])

                # If two borders of the grayscale image have near-black pixels, it is "probably" Dark Corner
                if ((sx_border and dx_border) or (up_border and down_border)
                or (sx_border and up_border) or (dx_border and down_border)
                or (sx_border and down_border) or (dx_border and up_border)) < DARK_THRESH:
                    
                    results = artifact.get_max_contour(mask)
                    if results is not None:
                        final_image, _, _ = results
                        final_image = cv2.resize(final_image, (HIGH_RES, HIGH_RES), interpolation=cv2.INTER_CUBIC)

                        cv2.imwrite(os.path.join(root_folder_path,new_image_folder,image_filename), final_image)
                
                ### If not Dark Corner, check for Shape Handling ###
                # If images is squared-shape, then resize
                elif height == width:
                    if (height > HIGH_RES):
                        final_image = cv2.resize(image_gt, (HIGH_RES, HIGH_RES), interpolation=cv2.INTER_AREA) # shrinking
                    elif (height < HIGH_RES): 
                        final_image = cv2.resize(image_gt, (HIGH_RES, HIGH_RES), interpolation=cv2.INTER_CUBIC) # enlarge
                    else:
                        final_image = image_gt
                    
                    cv2.imwrite(os.path.join(root_folder_path,new_image_folder,image_filename), final_image)

                else:
                    ### If neither Dark Corner or squared, then it is rectangular ###
                    # If height-width ratio is higher than a threshold -> split image into 3 pieces:
                    if (max(height, width) - min(height, width)) >= THRESH_RATIO:
                        image_a, image_b, image_c = artifact.rect2square_image_splitting(resize_dim=HIGH_RES)

                        new_image_filename_a = f'{id_image}_a.png'
                        new_image_filename_b = f'{id_image}_b.png'
                        new_image_filename_c = f'{id_image}_c.png'

                        cv2.imwrite(os.path.join(root_folder_path,new_image_folder,new_image_filename_a), image_a)
                        cv2.imwrite(os.path.join(root_folder_path,new_image_folder,new_image_filename_b), image_b)
                        cv2.imwrite(os.path.join(root_folder_path,new_image_folder,new_image_filename_c), image_c)


                    else:
                        center_image = artifact.center_crop(min(height, width), min(height, width))
                        final_image = cv2.resize(center_image, (HIGH_RES, HIGH_RES), interpolation=cv2.INTER_CUBIC)  # enlarge

                        cv2.imwrite(os.path.join(root_folder_path,new_image_folder,image_filename), final_image)

            except Exception: # NB: not an ideal solution to handle errors
                continue




if __name__ == "__main__":
    test()