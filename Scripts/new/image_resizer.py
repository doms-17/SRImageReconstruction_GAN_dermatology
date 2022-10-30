import os
import cv2
from tqdm import tqdm
import json


HIGH_RES = 1024
LOW_RES_256 = HIGH_RES // 4
LOW_RES_512 = HIGH_RES // 2
COMPRESSION = 6


def create_folder(path, sub_dir, folder_name):
    path = os.getcwd()
    # dir_name = pathlib.Path('/my/directory').mkdir(parents=True, exist_ok=True)
    dir_name = os.path.join(path, sub_dir, folder_name)
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


class ImageResize:
    def __init__(self, path):
        self.path = path
        # self.files = os.listdir(path)

        self.low_dir = "lowResolution"
        self.high_dir = "highResolution"
        self.high_plus_dir = "highResolution_plus"

        self.files_low = os.listdir(os.path.join(self.path, self.low_dir))
        self.files_high = os.listdir(os.path.join(self.path, self.high_dir))
        self.files_high_plus = os.listdir(
            os.path.join(self.path, self.high_plus_dir))

    def data_paired_resized(self):
        # root_and_dir = os.path.join(self.path)  # retrieve path of preDataset

        # retrieve path of preDataset
        root_and_dir_low = os.path.join(self.path, self.low_dir)
        root_and_dir_high = os.path.join(
            self.path, self.high_dir)  # retrieve path of preDataset
        root_and_dir_high_plus = os.path.join(
            self.path, self.high_plus_dir)  # retrieve path of preDataset

        new_root_and_dir_low_256 = create_folder(
            path="", sub_dir="dataset_paired_validationSet_resize", folder_name="lowResolution_256")  # create folder per each label
        new_root_and_dir_low_512 = create_folder(
            path="", sub_dir="dataset_paired_validationSet_resize", folder_name="lowResolution_512")  # create folder per each label
        # new_root_and_dir_high_512 = create_folder(
        #     path="", sub_dir="dataset_paired_sliding_resized", folder_name="highResolution_512")  # create folder per each label
        new_root_and_dir_high_plus_512 = create_folder(
            path="", sub_dir="dataset_paired_validationSet_resize", folder_name="highResolution_plus_512")  # create folder per each label

        for file_low, file_high, file_high_plus in tqdm(zip(self.files_low, self.files_high, self.files_high_plus)):
            image_low = cv2.imread(os.path.join(root_and_dir_low, file_low))
            # image_high = cv2.imread(os.path.join(root_and_dir_high, file_high))
            image_high_plus = cv2.imread(os.path.join(
                root_and_dir_high_plus, file_high_plus))

            image_low_resized_256 = cv2.resize(
                image_low, (LOW_RES_256, LOW_RES_256), cv2.INTER_AREA)
            image_low_resized_512 = cv2.resize(
                image_low, (LOW_RES_512, LOW_RES_512), cv2.INTER_AREA)
            image_high_plus_resized_512 = cv2.resize(
                image_high_plus, (LOW_RES_512, LOW_RES_512), cv2.INTER_AREA)

            new_img_filename_low_256 = os.path.join(
                new_root_and_dir_low_256, file_low)
            new_img_filename_low_512 = os.path.join(
                new_root_and_dir_low_512, file_low)
            new_img_filename_high_plus_512 = os.path.join(
                new_root_and_dir_high_plus_512, file_high_plus)

            cv2.imwrite(new_img_filename_low_256, image_low_resized_256, [
                        cv2.IMWRITE_PNG_COMPRESSION, COMPRESSION],)
            cv2.imwrite(new_img_filename_low_512, image_low_resized_512, [
                        cv2.IMWRITE_PNG_COMPRESSION, COMPRESSION],)
            # cv2.imwrite(new_img_filename_high_512, image_high_resized_512, [
            #             cv2.IMWRITE_PNG_COMPRESSION, COMPRESSION],)
            cv2.imwrite(new_img_filename_high_plus_512, image_high_plus_resized_512, [
                        cv2.IMWRITE_PNG_COMPRESSION, COMPRESSION],)


def test():
    dataset_name = "dataset_paired_validationSet"
    newDataset_name = "dataset_paired_validationSet_resize"
    root_rawDataset = f"{os.getcwd()}\\{dataset_name}"

    # low_dir_x8_res_x4 = create_folder(path="", sub_dir=newDataset_name, folder_name="lowResolution_x8_res_x4")
    # low_dir_x8_res_x2 = create_folder(path="", sub_dir=newDataset_name, folder_name="lowResolution_x8_res_x2")
    # low_dir_x4_res_x4 = create_folder(path="", sub_dir=newDataset_name, folder_name="lowResolution_x4_res_x4")
    # low_dir_x4_res_x2 = create_folder(path="", sub_dir=newDataset_name, folder_name="lowResolution_x4_res_x2")
    # high_dir_plus_res_x2 = create_folder(path="", sub_dir=newDataset_name, folder_name="highResolution_plus_res_x2")

    dataset = ImageResize(path=root_rawDataset)
    dataset.data_paired_resized()


if __name__ == "__main__":
    test()
