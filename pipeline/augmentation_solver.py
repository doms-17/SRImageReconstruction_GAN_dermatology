import os
import random

import cv2
import albumentations as A
import sklearn
from matplotlib import pyplot as plt
from tqdm import tqdm



def create_folder(dir_name):
    os.makedirs(dir_name, exist_ok=True)
    return dir_name



class Augmentation:
    def __init__(self, image):
        self.image = image

    def augment(self, vertFlip=0, horFlip=0):
        transform = A.Compose([
            A.Affine(scale=(1.05), keep_ratio=True, shear=[-5,5], 
                    interpolation=cv2.INTER_CUBIC, mode=cv2.BORDER_CONSTANT, p=0.5),
            A.VerticalFlip(p=vertFlip),
            A.HorizontalFlip(p=horFlip),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.2, p=1),
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
                A.CenterCrop(height=self.image.shape[0]-50, width=self.image.shape[1]-50, p=1),
                A.Resize(height=self.image.shape[0], width=self.image.shape[1],
                         interpolation=cv2.INTER_CUBIC, p=1),
            ], p=1),
        ], p=1)
        return transform(image=self.image)['image']



def main():
    root_dataset: str = "D:/DOMI/University/Thesis/Coding/Dataset/Classification/SR/CC_ISIC_not1024_label"
    labels: list[str] = os.listdir(root_dataset)

    for label in labels:
        all_filenames_per_label: list[str] = os.listdir(os.path.join(root_dataset,label))
        if label == "AKIEC":
            all_filenames_per_label, _ = sklearn.model_selection.train_test_split(
                                    all_filenames_per_label, train_size=0.9, random_state=1)
        elif label == "KL":
            all_filenames_per_label, _ = sklearn.model_selection.train_test_split(
                                    all_filenames_per_label, train_size=0.16, random_state=1)

        for image_filename in tqdm(all_filenames_per_label):
            image_ID = image_filename.split(".")[0]
            image_ext = image_filename.split(".")[1]
            image = cv2.imread(os.path.join(root_dataset,label,image_filename))
            if label == "AKIEC":
                augmented_img_1 = Augmentation(image).augment(vertFlip=1)
                # plot_images(cv2.cvtColor(image,cv2.COLOR_BGR2RGB), cv2.cvtColor(augmented_img_1,cv2.COLOR_BGR2RGB))
                new_img_filename = os.path.join(root_dataset,label,image_ID+"_aug1."+image_ext)
                cv2.imwrite(new_img_filename, augmented_img_1, [cv2.IMWRITE_PNG_COMPRESSION, 6])

                augmented_img_2 = Augmentation(image).augment(horFlip=1)
                # plot_images(cv2.cvtColor(image,cv2.COLOR_BGR2RGB), cv2.cvtColor(augmented_img_2,cv2.COLOR_BGR2RGB))
                new_img_filename = os.path.join(root_dataset,label,image_ID+"_aug2."+image_ext)
                cv2.imwrite(new_img_filename, augmented_img_2, [cv2.IMWRITE_PNG_COMPRESSION, 6])

                augmented_img_3 = Augmentation(image).augment(vertFlip=1, horFlip=1)
                # plot_images(cv2.cvtColor(image,cv2.COLOR_BGR2RGB), cv2.cvtColor(augmented_img_3,cv2.COLOR_BGR2RGB))
                new_img_filename = os.path.join(root_dataset,label,image_ID+"_aug3."+image_ext)
                cv2.imwrite(new_img_filename, augmented_img_3, [cv2.IMWRITE_PNG_COMPRESSION, 6])

            elif label == "BCC":
                augmented_img_1 = Augmentation(image).augment(vertFlip=1)
                new_img_filename = os.path.join(root_dataset,label,image_ID+"_aug1."+image_ext)
                cv2.imwrite(new_img_filename, augmented_img_1, [cv2.IMWRITE_PNG_COMPRESSION, 6])

                augmented_img_2 = Augmentation(image).augment(horFlip=1)
                new_img_filename = os.path.join(root_dataset,label,image_ID+"_aug2."+image_ext)
                cv2.imwrite(new_img_filename, augmented_img_2, [cv2.IMWRITE_PNG_COMPRESSION, 6])
            
            elif label == "KL":
                augmented_img_1 = Augmentation(image).augment(vertFlip=1)
                # plot_images(cv2.cvtColor(image,cv2.COLOR_BGR2RGB), cv2.cvtColor(augmented_img_1,cv2.COLOR_BGR2RGB))
                new_img_filename = os.path.join(root_dataset,label,image_ID+"_aug1."+image_ext)
                cv2.imwrite(new_img_filename, augmented_img_1, [cv2.IMWRITE_PNG_COMPRESSION, 6])




if __name__ == "__main__":
    main()