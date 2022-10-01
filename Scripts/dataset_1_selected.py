import os
import json
from tqdm import tqdm

def create_folder(path, sub_dir, folder_name):
    path = os.getcwd()
    # dir_name = pathlib.Path('/my/directory').mkdir(parents=True, exist_ok=True) 
    dir_name = os.path.join(path, sub_dir, folder_name)
    os.makedirs(dir_name, exist_ok=True)
    return dir_name

class ImageSelected:
    def __init__(self, path_original, path_toCheck):
        self.path_original = path_original
        self.path_toCheck = path_toCheck
        self.labels = os.listdir(path_toCheck)
        self.files_per_label = {label:[] for label in self.labels}
        self.len_per_label = {label:0 for label in self.labels}

    def selected(self):
        # for idx, label in enumerate(self.labels):
        for label in self.labels:
            files_toCheck = os.listdir(os.path.join(self.path_toCheck, label))
            self.files_per_label[label] += files_toCheck
            self.len_per_label[label] += len(files_toCheck)
        return self.files_per_label, self.len_per_label

    def not_selected(self):
        for label in self.labels:
            files_original = os.listdir(os.path.join(self.path_original, label))
            files_toCheck = os.listdir(os.path.join(self.path_toCheck, label))
            deleted_files = set(files_original) ^ set(files_toCheck)

            self.files_per_label[label] += deleted_files
            self.len_per_label[label] += len(deleted_files)
        return self.files_per_label, self.len_per_label


def test():
    dataset_name_original = "dataset_noArtifact_or"
    dataset_name_toCheck = "dataset_noArtifact"
    root_orDataset = f"{os.getcwd()}\\{dataset_name_original}"
    root_rawDataset = f"{os.getcwd()}\\{dataset_name_toCheck}"

    dataset_manually_selected = ImageSelected(path_original=root_orDataset, path_toCheck=root_rawDataset)
    files_per_label, len_per_label = dataset_manually_selected.selected()
    dataset_deleted = ImageSelected(path_original=root_orDataset, path_toCheck=root_rawDataset)
    files_deleted_per_label, len_deleted_per_label = dataset_deleted.not_selected()

    info_selected = {"Pipeline":"Images manually selected", "len_per_label":len_per_label, "files_per_label":files_per_label}
    info_deleted = {"Pipeline":"Images deleted", "len_per_label":len_deleted_per_label, "files_per_label":files_deleted_per_label}
    filename_savings_selected = 'files_selected_per_label.json'
    filename_savings_deleted = 'files_deleted_per_label.json'
    json.dump(info_selected, open(filename_savings_selected, 'w'), indent=3)
    json.dump(info_deleted, open(filename_savings_deleted, 'w'), indent=3)

if __name__ == "__main__":
    test()
    