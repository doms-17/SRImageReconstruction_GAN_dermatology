
import os
import cv2
import albumentations as A
import random
import json
from tqdm import tqdm


HIGH_RES = 1024
LOW_RES = HIGH_RES // 4
COMPRESSION = 6


def create_folder(path, sub_dir, folder_name):
    path = os.getcwd()
    # dir_name = pathlib.Path('/my/directory').mkdir(parents=True, exist_ok=True)
    dir_name = os.path.join(path, sub_dir, folder_name)
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


class Augmentation:
    def __init__(self, image):
        self.image = image

    def augment(self):
        transform = A.Compose([
            A.Affine(scale=(1.05), keep_ratio=True,
                     shear=[-5, 5], interpolation=cv2.INTER_CUBIC, mode=cv2.BORDER_CONSTANT, p=0.5),
            A.Flip(p=1),
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.2, p=1),
                A.RandomGamma(gamma_limit=(60, 140), p=1),
            ], p=1),
            A.OneOf([
                A.ElasticTransform(alpha=500, sigma=50, alpha_affine=10,
                                   interpolation=cv2.INTER_CUBIC, border_mode=cv2.BORDER_CONSTANT, p=1),
                A.GridDistortion(num_steps=20, distort_limit=0.05,
                                 interpolation=cv2.INTER_CUBIC, border_mode=cv2.BORDER_CONSTANT, p=1),
                A.NoOp(p=1),
            ], p=1),
            A.Compose([
                A.CenterCrop(height=HIGH_RES-50, width=HIGH_RES-50, p=1),
                A.Resize(height=HIGH_RES, width=HIGH_RES,
                         interpolation=cv2.INTER_CUBIC, p=1),
            ], p=1),
        ], p=1)
        return transform(image=self.image)['image']


class Paired:
    def __init__(self, image):
        self.image = image

    def lowres_2order(self, scale_1, scale_2):
        lowres_1st_2nd = A.Compose([
            ########## 1ST ORDER ##########
            # -----Blur
            A.OneOf([
                A.GaussianBlur(blur_limit=(7, 21),
                               sigma_limit=(0.2, 3), p=0.7),
                A.AdvancedBlur(blur_limit=(7, 21), sigmaX_limit=(
                    0.2, 3), sigmaY_limit=(0.2, 3), beta_limit=(0.5, 4), p=0.15),
                A.RingingOvershoot(blur_limit=(7, 21), p=0.1),
            ], p=1),
            # -----Downscale
            A.OneOf([
                # A.Downscale(scale_min=1/LOW_SCALE, scale_max=1/LOW_SCALE, interpolation=cv2.INTER_NEAREST, p=1),
                A.Downscale(scale_min=1/scale_1, scale_max=1/scale_1,
                            interpolation=cv2.INTER_LINEAR, p=1),
                A.Downscale(scale_min=1/scale_1, scale_max=1/scale_1,
                            interpolation=cv2.INTER_AREA, p=1),
                A.Downscale(scale_min=1/scale_1, scale_max=1/scale_1,
                            interpolation=cv2.INTER_CUBIC, p=1),
            ], p=1),
            # -----Noise
            A.OneOf([
                A.GaussNoise(var_limit=(1, 30), per_channel=True, p=0.5),
                A.GaussNoise(var_limit=(1, 30), per_channel=False, p=0.4),
                A.ISONoise(intensity=(0.05, 0.5), p=0.5),
            ], p=1),
            # -----Compression
            A.ImageCompression(quality_lower=30, quality_upper=95, p=1),

            ########## 2ND ORDER ##########
            # -----Blur
            A.OneOf([
                A.GaussianBlur(blur_limit=(7, 21),
                               sigma_limit=(0.2, 3), p=0.7),
                A.AdvancedBlur(blur_limit=(7, 21), sigmaX_limit=(
                    0.2, 3), sigmaY_limit=(0.2, 3), beta_limit=(0.5, 4), p=0.15),
                A.RingingOvershoot(blur_limit=(7, 21), p=0.1),
            ], p=0.8),
            # -----Downscale
            A.OneOf([
                # A.Downscale(scale_min=1/scale, scale_max=1/scale, interpolation=cv2.INTER_NEAREST, p=1),
                A.Downscale(scale_min=1/scale_2, scale_max=1/scale_2,
                            interpolation=cv2.INTER_LINEAR, p=1),
                A.Downscale(scale_min=1/scale_2, scale_max=1/scale_2,
                            interpolation=cv2.INTER_AREA, p=1),
                A.Downscale(scale_min=1/scale_2, scale_max=1/scale_2,
                            interpolation=cv2.INTER_CUBIC, p=1),
            ], p=1),
            # -----Noise
            A.OneOf([
                A.GaussNoise(var_limit=(1, 25), per_channel=True, p=0.5),
                A.GaussNoise(var_limit=(1, 30), per_channel=False, p=0.4),
                A.ISONoise(intensity=(0.05, 0.1), p=0.5),
            ], p=1),
            # -----Compression
            A.OneOf([
                A.Compose([
                    A.ImageCompression(
                        quality_lower=30, quality_upper=95, p=1),
                    A.RingingOvershoot(blur_limit=(7, 21), p=0.8),
                ], p=1),
                A.Compose([
                    A.RingingOvershoot(blur_limit=(7, 21), p=0.8),
                    A.ImageCompression(
                        quality_lower=30, quality_upper=95, p=1),
                ], p=1),
            ], p=1),
        ], p=1)
        return lowres_1st_2nd(image=self.image)['image']

    def highres_sharp(self):
        highres_plus = A.Compose([
            A.Sharpen(alpha=(0.5, 0.5), lightness=(1, 1), p=1),
            A.MedianBlur(blur_limit=(3, 3), p=0.8),
        ], p=1)
        return highres_plus(image=self.image)['image']


class ImageTransform:
    def __init__(self, path):
        self.path = path
        self.labels = os.listdir(path)
        self.data_labels = {label: [] for label in self.labels}
        self.len_labels = {label: 0 for label in self.labels}

        # json_to_dict = json.load(open("D:\\DOMI\\University\\Magistrale\\Tesi\\Pipeline_coding\\file_manually_selected_per_label.json"))
        # self.files_manually_selected = json_to_dict["file_per_labels"]

        for label in self.labels:
            files = os.listdir(os.path.join(path, label))
            self.data_labels[label] += files
            self.len_labels[label] += len(files)

    def data_paired(self):
        for label in self.labels:
            root_and_dir_noArtifact = os.path.join(self.path, label)

            new_root_and_dir_low_x4 = create_folder(
                path="", sub_dir="dataset_paired_validationSet\\lowResolution", folder_name="")
            new_root_and_dir_high = create_folder(
                path="", sub_dir="dataset_paired_validationSet\\highResolution", folder_name="")
            new_root_and_dir_high_plus = create_folder(
                path="", sub_dir="dataset_paired_validationSet\\highResolution_plus", folder_name="")

            resize_transform = A.Compose([
                A.CenterCrop(height=HIGH_RES-5, width=HIGH_RES-5, p=1),
                A.Resize(height=HIGH_RES, width=HIGH_RES,
                         interpolation=cv2.INTER_CUBIC, p=1),
            ], p=1)

            files = os.listdir(root_and_dir_noArtifact)
            for file in tqdm(files):
                # if file in self.files_manually_selected[label]:
                image_start = cv2.imread(
                    os.path.join(root_and_dir_noArtifact, file))
                image = resize_transform(image=image_start)['image']
                try:
                    # augmentation = Augmentation(image)

                    # augmented_image = augmentation.augment()
                    # id_image = file.split('.')[0]
                    # aug_file = f'{id_image}_aug.png'

                    paired = Paired(image)

                    #----- Paired Image -----#

                    rnd_scale_1 = random.randint(4, 8)
                    rnd_scale_2 = random.randint(4, 8)

                    lowres_image_x4 = paired.lowres_2order(
                        scale_1=rnd_scale_1, scale_2=rnd_scale_2)
                    highres_image_plus = paired.highres_sharp()

                    # paired_augmented = Paired(augmented_image)

                    #----- Paired Augmented -----#
                    # rnd_scale_1 = random.randint(4, 8)
                    # rnd_scale_2 = random.randint(4, 8)

                    # lowres_image_aug_x4 = paired_augmented.lowres_2order(
                    #     scale_1=rnd_scale_1, scale_2=rnd_scale_2)
                    # highres_image_aug_plus = paired_augmented.highres_sharp()

                    #----- Save paired -----#
                    new_img_filename_low_x4 = os.path.join(
                        new_root_and_dir_low_x4, file)
                    cv2.imwrite(new_img_filename_low_x4, lowres_image_x4, [
                                cv2.IMWRITE_PNG_COMPRESSION, COMPRESSION])
                    new_img_filename_high = os.path.join(
                        new_root_and_dir_high, file)
                    cv2.imwrite(new_img_filename_high, image, [
                                cv2.IMWRITE_PNG_COMPRESSION, COMPRESSION])
                    new_img_filename_high_plus = os.path.join(
                        new_root_and_dir_high_plus, file)
                    cv2.imwrite(new_img_filename_high_plus, highres_image_plus, [
                                cv2.IMWRITE_PNG_COMPRESSION, COMPRESSION])

                    #----- Save augmented_paired -----#
                    # new_img_filename_low_x4 = os.path.join(
                    #     new_root_and_dir_low_x4, aug_file)
                    # cv2.imwrite(new_img_filename_low_x4, lowres_image_aug_x4, [
                    #             cv2.IMWRITE_PNG_COMPRESSION, COMPRESSION])
                    # new_img_filename_high = os.path.join(
                    #     new_root_and_dir_high, aug_file)
                    # cv2.imwrite(new_img_filename_high, augmented_image, [
                    #             cv2.IMWRITE_PNG_COMPRESSION, COMPRESSION])
                    # new_img_filename_high_plus = os.path.join(
                    #     new_root_and_dir_high_plus, aug_file)
                    # cv2.imwrite(new_img_filename_high_plus, highres_image_aug_plus, [
                    #             cv2.IMWRITE_PNG_COMPRESSION, COMPRESSION])

                except Exception:
                    continue


def test():
    dataset_name = "to_test_discarded"
    newDataset_name = "dataset_paired_validationSet"
    root_rawDataset = f"{os.getcwd()}\\{dataset_name}"

    # low_dir_x8 = create_folder(
    #     path="", sub_dir=newDataset_name, folder_name="lowResolution_x8")
    low_dir_x4 = create_folder(
        path="", sub_dir=newDataset_name, folder_name="lowResolution")
    high_dir = create_folder(
        path="", sub_dir=newDataset_name, folder_name="highResolution")
    high_dir_plus = create_folder(
        path="", sub_dir=newDataset_name, folder_name="highResolution_plus")

    dataset = ImageTransform(path=root_rawDataset)
    dataset.data_paired()


if __name__ == "__main__":
    test()
