import os
from tqdm import tqdm

def main():
    dir_name = "D:\\DOMI\\University\\Magistrale\\Tesi\\Pipeline_coding\\dataset_paired_new"
    subdirs_name = os.listdir(dir_name)

    for subdir_name in subdirs_name:
        path = f'{dir_name}\\{subdir_name}'
        files = os.listdir(path)
        for file in tqdm(files):
            id_image = file.split('.')[0]
            if id_image.endswith("_a"):
                os.remove(os.path.join(path, file))

if __name__ == "__main__":
    main()