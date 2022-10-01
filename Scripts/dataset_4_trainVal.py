import os
import json
import shutil
from tqdm import tqdm


def create_folder(path, sub_dir, folder_name):
    path = os.getcwd()
    dir_name = os.path.join(path, sub_dir, folder_name)
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


class ValidSet:
    def __init__(self, path, path_json, filename_json):
        self.path = path
        self.path_json = path_json
        self.filename_json = filename_json

        json_to_dict = json.load(open(os.path.join(self.path_json, self.filename_json)))
        self.files_deleted_per_label = json_to_dict["files_per_label"]
        self.labels = self.files_deleted_per_label.keys()

    def split(self, valFolder):
        for label in self.labels:
            root_and_dir_noArtifact_or = os.path.join(self.path, label)
            new_root_and_dir = create_folder(path="", sub_dir="validSet_per_label", folder_name=label)
            files = os.listdir(root_and_dir_noArtifact_or)
            for file in tqdm(files):
                if file in self.files_deleted_per_label[label]:
                    shutil.copyfile(os.path.join(root_and_dir_noArtifact_or,file), os.path.join(new_root_and_dir,file))
                    

def test():
    dataset_name = "dataset_noArtifact_or"
    root_or = f"{os.getcwd()}\\{dataset_name}"

    path_json = os.getcwd()
    filename_json = 'files_deleted_per_label.json'

    newDataset_name = "validSet_per_label"
    validSet_folder = create_folder(path="", sub_dir=newDataset_name, folder_name="")

    validSet = ValidSet(path=root_or, path_json=path_json, filename_json=filename_json)
    validSet.split(validSet_folder)


if __name__ == "__main__":
    test()