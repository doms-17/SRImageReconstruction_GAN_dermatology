import os
import shutil
from tqdm import tqdm

def create_folder(dir_name: str) -> str:
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


def test() -> None:
    root_path: str = ""
    new_root_path: str = ""

    sub_folders: list[str] = [f for f in os.listdir(root_path) if not f.startswith('.')]

    for folder in sub_folders:
        files: list[str] = [f for f in os.listdir(os.path.join(root_path,folder)) if not f.startswith('.')]
        for file in tqdm(files):
            # iteration: str = file.split('_')[-1].split('.')[0]
            # new_folder: str = create_folder(os.path.join(new_root_path,iteration))
            shutil.copyfile(os.path.join(root_path,folder,file), os.path.join(new_root_path,file))

if __name__ == "__main__":
    test()
