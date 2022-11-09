import os
import json
import shutil
import cv2
from tqdm import tqdm


def create_folder(dir_name: str):
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


def test():
    root_dataset_or: str = "D:\\DOMI\\University\\Magistrale\\Tesi\\Pipeline_coding\\Original_orsize"
    root_dataset_no_artifact: str = "D:\\DOMI\\University\\Magistrale\\Tesi\\Pipeline_coding\\dataset_noArtifact"

    new_root_dataset: str = "D:\\DOMI\\University\\Magistrale\\Tesi\\Pipeline_coding\\Test Set\\to_test_not1024"

    labels: list[str] = os.listdir(root_dataset_or)
    images_not1024_deleted_per_label: dict = {}
    images_not1024_per_label: dict = {'AKIEC': [], 'BCC': [
    ], 'DF': [], 'KL': [], 'MEL': [], 'NV': [], 'VASC': []}

    for label in labels:
        images_not1024_deleted_temp = []
        files_or: list[str] = os.listdir(os.path.join(root_dataset_or, label))
        files_no_artifact: list[str] = os.listdir(
            os.path.join(root_dataset_no_artifact, label))
        images_not1024_deleted_temp += set(
            files_or) ^ set(files_no_artifact)
        images_not1024_deleted_per_label[label] = images_not1024_deleted_temp
    # filename_images_not1024_notselected: str = "list_of_images_not1024_notselected.json"
    # json.dump(images_not1024_notselected, open(filename_images_not1024_notselected, 'w'), indent=3)

    files_manually_deleted = json.load(open(
        "files_deleted_per_label(valSet).json"))['files_per_label']

    for label in labels:
        for file in tqdm(images_not1024_deleted_per_label[label]):
            if ((file not in files_manually_deleted[label]) and ('rez' not in file.split('.')[0])):
                image = cv2.imread(os.path.join(root_dataset_or, label, file))
                height, width, _ = image.shape
                if (height < 1024 and width < 1024):
                    images_not1024_per_label[label].append(file)
    json.dump(images_not1024_per_label, open(
        "list_of_images_not1024.json", 'w'), indent=3)

    # shutil.copyfile(os.path.join(root_dataset_or, label,
    #                 file), os.path.join(new_root_dataset, file),)


if __name__ == "__main__":
    test()
