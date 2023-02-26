import os
import shutil
import json
import random

import sklearn
from tqdm import tqdm



TRAIN: float = 0.75
VAL: float = 0.10
TEST: float = 0.15



def create_folder(dir_name):
    os.makedirs(dir_name, exist_ok=True)
    return dir_name



def main():
    root_dataset: str = "D:/DOMI/University/Thesis/Coding/Dataset/Classification/SR/CC_ISIC_not1024_label"
    root_dataset_dest: str = "D:/DOMI/University/Thesis/Coding/Dataset/Classification/SR/CC_ISIC_not1024_trainValTest"
    create_folder(root_dataset_dest)

    json_filename_dataset_split:str = f"{root_dataset_dest}/dataset_split_filenames.json"

    create_folder(os.path.join(root_dataset_dest,"train"))
    create_folder(os.path.join(root_dataset_dest,"val"))
    create_folder(os.path.join(root_dataset_dest,"test"))
    labels: list[str] = os.listdir(root_dataset)
    dataset_split = {}

    for label in tqdm(labels):
        create_folder(os.path.join(root_dataset_dest,"train",label))
        create_folder(os.path.join(root_dataset_dest,"val",label))
        create_folder(os.path.join(root_dataset_dest,"test",label))
        all_filenames_per_label: list[str] = os.listdir(os.path.join(root_dataset,label))

        X_train, X_test = sklearn.model_selection.train_test_split(all_filenames_per_label, test_size=TEST, random_state=1)
        X_train, X_val = sklearn.model_selection.train_test_split(X_train, test_size=VAL, random_state=1)

        dataset_split[label] = {"train": X_train, "val": X_val, "test": X_test}

        for train_filename in X_train:
            shutil.copyfile(os.path.join(root_dataset, label, train_filename),
                            os.path.join(root_dataset_dest, "train", label, train_filename),)
        for val_filename in X_val:
            shutil.copyfile(os.path.join(root_dataset, label, val_filename),
                            os.path.join(root_dataset_dest, "val", label, val_filename),)
        for test_filename in X_test:
            shutil.copyfile(os.path.join(root_dataset, label, test_filename),
                            os.path.join(root_dataset_dest, "test", label, test_filename),)

    json.dump(dataset_split, open(json_filename_dataset_split, 'w'), indent=3)




if __name__ == "__main__":
    main()