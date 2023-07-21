import os
import shutil
import json
from tqdm import tqdm


def main():
    # root_dataset_Or: str = ""
    # root_dataset_CC: str = ""
    # root_dataset_dest: str = ""

    # imagesOr_filenames = os.listdir(root_dataset_Or)

    # labels:str = os.listdir(root_dataset_CC)

    # for imageOr_filename in tqdm(imagesOr_filenames):
    #     for label in labels:
    #         imagesCC_filenames = os.listdir(os.path.join(root_dataset_CC,label))
    #         if imageOr_filename in imagesCC_filenames:
    #             shutil.copyfile(os.path.join(root_dataset_CC,label,imageOr_filename), os.path.join(root_dataset_dest,imageOr_filename))

    root_dataset_Or: str = ""
    root_dataset_CC: str = ""
    root_dataset_dest: str = ""

    imagesOr_filenames = os.listdir(root_dataset_Or)
    imagesCC_filenames = os.listdir(os.path.join(root_dataset_CC))

    # labels:str = os.listdir(root_dataset_CC)

    for imageOr_filename in tqdm(imagesOr_filenames):
        if imageOr_filename in imagesCC_filenames:
            shutil.copyfile(os.path.join(root_dataset_Or,imageOr_filename), os.path.join(root_dataset_dest,imageOr_filename))


if __name__ == "__main__":
    main()
