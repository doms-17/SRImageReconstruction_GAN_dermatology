import os
import shutil
import json
from tqdm import tqdm


def main():
    root_dataset_or: str = "D:\\DOMI\\University\\Magistrale\\Tesi\\Pipeline_coding\\Original_orsize"
    root_dataset_dest: str = "D:\\DOMI\\University\\Magistrale\\Tesi\\Pipeline_coding\\Test Set\\to_test_not1024"

    files_testSet = json.load(open(
        "files_not1024_pet_label(testSet).json"))

    labels: list[str] = os.listdir(root_dataset_or)

    for label in labels:
        for file in tqdm(files_testSet[label]):
            shutil.copyfile(os.path.join(root_dataset_or, label,
                            file), os.path.join(root_dataset_dest, label, file),)


if __name__ == "__main__":
    main()
