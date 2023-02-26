import os
import shutil
import json
from tqdm import tqdm


def main():
    # root_dataset_Or: str = "D:/DOMI/University/Thesis/Coding/Dataset/TestSet/ISIC_not1024"
    # root_dataset_CC: str = "D:/DOMI/University/Thesis/Coding/Dataset/GANNED_orsize_selected/"
    # root_dataset_dest: str = "D:/DOMI/University/Thesis/Coding/Dataset/TestSet/CC_ISIC_not1024/"

    # imagesOr_filenames = os.listdir(root_dataset_Or)

    # labels:str = os.listdir(root_dataset_CC)

    # for imageOr_filename in tqdm(imagesOr_filenames):
    #     for label in labels:
    #         imagesCC_filenames = os.listdir(os.path.join(root_dataset_CC,label))
    #         if imageOr_filename in imagesCC_filenames:
    #             shutil.copyfile(os.path.join(root_dataset_CC,label,imageOr_filename), os.path.join(root_dataset_dest,imageOr_filename))

    root_dataset_Or: str = "D:/DOMI/University/Thesis/Coding/Dataset/Inference/derma_v0/ISIC_not1024"
    root_dataset_CC: str = "D:/DOMI/University/Thesis/Coding/Dataset/Inference/derma_v0/CC_ISIC_not1024"
    root_dataset_dest: str = "D:/DOMI/University/Thesis/Coding/Dataset/Inference/derma_v0/ISIC_not1024_new"

    imagesOr_filenames = os.listdir(root_dataset_Or)
    imagesCC_filenames = os.listdir(os.path.join(root_dataset_CC))

    # labels:str = os.listdir(root_dataset_CC)

    for imageOr_filename in tqdm(imagesOr_filenames):
        if imageOr_filename in imagesCC_filenames:
            shutil.copyfile(os.path.join(root_dataset_Or,imageOr_filename), os.path.join(root_dataset_dest,imageOr_filename))


if __name__ == "__main__":
    main()