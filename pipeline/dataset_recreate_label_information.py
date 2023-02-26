import os
import shutil
import json
from tqdm import tqdm

def main():
    json_filename:str = "D:/DOMI/University/Thesis/Coding/Dataset/files_not1024_per_label.json"
    root_dataset_or: str = "D:/DOMI/University/Thesis/Coding/Dataset/TestSet/ISIC_not1024"
    root_dataset_dest: str = "D:/DOMI/University/Thesis/Coding/Dataset/TestSet/ISIC_not1024_label"

    labels: list[str] = os.listdir(root_dataset_dest)
    images_or_filenames: list[str] = os.listdir(root_dataset_or)
    files_per_labels: dict[str: list] = json.load(open(json_filename))

    for image_filename in tqdm(images_or_filenames):
        for label in labels:
            id_image: str = f"{image_filename.split('.')[0].split('_out')[0]}.{image_filename.split('.')[-1]}"
            if id_image in files_per_labels[label]:
                shutil.copyfile(os.path.join(root_dataset_or, image_filename),
                                os.path.join(root_dataset_dest, label, image_filename),)


if __name__ == "__main__":
    main()