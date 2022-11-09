import os
import random
import json
import shutil
from tqdm import tqdm

TRAIN: float = 0.7
VAL: float = 0.1
TEST: float = 0.2


def create_folder(path, sub_dir="", folder_name=""):
    path = os.getcwd()
    # dir_name = pathlib.Path('/my/directory').mkdir(parents=True, exist_ok=True)
    dir_name = os.path.join(path, sub_dir, folder_name)
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


def create_setFolders(main_path):
    root_trainSet_low = create_folder(
        path=f"{main_path}\\lowResolution\\train")
    root_valSet_low = create_folder(
        path=f"{main_path}\\lowResolution\\val")
    root_testSet_low = create_folder(
        path=f"{main_path}\\lowResolution\\test")

    root_trainSet_high_plus = create_folder(
        path=f"{main_path}\\highResolution_plus\\train")
    root_valSet_high_plus = create_folder(
        path=f"{main_path}\\highResolution_plus\\val")
    root_testSet_high_plus = create_folder(
        path=f"{main_path}\\highResolution_plus\\test")
    return root_trainSet_low


def val_set(root_allFiles):
    all_files_name = os.listdir(root_allFiles)
    val_files_name = random.sample(
        all_files_name, int(len(all_files_name)*VAL))
    val_files_name_json = 'val_files_name.json'
    json.dump(val_files_name, open(val_files_name_json, 'w'), indent=3)


def test_set(root_allFiles):
    all_files_name = os.listdir(root_allFiles)
    test_files_name = random.sample(
        all_files_name, int(len(all_files_name)*TEST))
    test_files_name_json = 'test_files_name.json'
    json.dump(test_files_name, open(test_files_name_json, 'w'), indent=3)


def train_set(root_allFiles):
    train_files_name = os.listdir(root_allFiles)
    train_files_name_json = 'train_files_name.json'
    json.dump(train_files_name, open(train_files_name_json, 'w'), indent=3)


def move_files(orig_folder, dest_folder, jsonfile):
    jsonToList = json.load(open(jsonfile))

    for file in tqdm(jsonToList):
        # shutil.move(os.path.join(root_train_set_low, file), os.path.join(root_val_set_low, file),)
        shutil.move(os.path.join(orig_folder, file),
                    os.path.join(dest_folder, file),)


if __name__ == "__main__":
    # path = "D:\\DOMI\\University\\Magistrale\\Tesi\\Pipeline_coding\\dataset_paired_sliding_resized_new"
    # root_trainSet_low = create_setFolders(path)

    root_all_files = "D:\\DOMI\\University\\Magistrale\\Tesi\\Pipeline_coding\\dataset_paired_sliding_new\\highResolution\\train"
    # dest_files = "D:\\DOMI\\University\\Magistrale\\Tesi\\Pipeline_coding\\dataset_paired_sliding_new\\highResolution\\test"
    # filejson = "D:\\DOMI\\University\\Magistrale\\Tesi\\Pipeline_coding\\test_files_name.json"
    # move_files(root_all_files, dest_files, filejson)
    train_set(root_all_files)
    # test_set(root_all_files)
    # val_set(root_all_files)
