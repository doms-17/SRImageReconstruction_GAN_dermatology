import os
import cv2
import pathlib
import albumentations as A
from tqdm import tqdm
from matplotlib import pyplot as plt

HIGH_RES = 1024
LOW_RES = HIGH_RES // 4
LOW_SCALE = 4

def create_folder(path, sub_dir, folder_name):
    path = os.getcwd()
    # dir_name = pathlib.Path('/my/directory').mkdir(parents=True, exist_ok=True) 
    dir_name = os.path.join(path, sub_dir, folder_name)
    os.makedirs(dir_name, exist_ok=True)
    return dir_name

def plot_img(img):
    plt.figure(figsize=(20,10))
    plt.imshow(img)
    plt.show()

def plot_images(img1, img2):
    plt.figure(figsize=(20,10))
    ax1 = plt.subplot(211)
    ax1.imshow(img1)
    ax2 = plt.subplot(212)
    ax2.imshow(img2)
    plt.show()


#####--------------------------Complex
augment = A.Compose([
    A.OneOf([
        A.HorizontalFlip(p=1),
        A.VerticalFlip(p=1),
        A.RandomRotate90(p=1),
    ], p=1),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.3, p=1),
        A.RandomGamma(gamma_limit=(60,150), p=1),
    ]),
    A.OneOf([
        A.ElasticTransform(alpha=1000, sigma=50, alpha_affine=10, interpolation=cv2.INTER_CUBIC, border_mode=cv2.BORDER_CONSTANT, p=1),
        A.GridDistortion(num_steps=20, distort_limit=0.3, interpolation=cv2.INTER_CUBIC, border_mode=cv2.BORDER_CONSTANT, p=1),
        A.NoOp(p=1)
    ],p=1)
])

lowres_first = A.Compose([
    #-----Blur
    A.OneOf([
        A.GaussianBlur(blur_limit=(7,21), sigma_limit=(0.2,3), p=0.7),
        A.AdvancedBlur(blur_limit=(7,21), sigmaX_limit=(0.2,3), sigmaY_limit=(0.2,3), beta_limit=(0.5,4), p=0.15),
        A.RingingOvershoot(blur_limit=(7,21), p=0.1)
        ], p=1),
    #-----Downscale
    A.OneOf([
        # A.Downscale(scale_min=1/LOW_SCALE, scale_max=1/LOW_SCALE, interpolation=cv2.INTER_NEAREST, p=1),
        A.Downscale(scale_min=1/LOW_SCALE, scale_max=1/LOW_SCALE, interpolation=cv2.INTER_LINEAR, p=1),
        A.Downscale(scale_min=1/LOW_SCALE, scale_max=1/LOW_SCALE, interpolation=cv2.INTER_AREA, p=1),
        A.Downscale(scale_min=1/LOW_SCALE, scale_max=1/LOW_SCALE, interpolation=cv2.INTER_CUBIC, p=1),
        ], p=1),
    #-----Noise
    A.OneOf([
        A.GaussNoise(var_limit=(1,30), p=0.5),
        A.GaussNoise(var_limit=(1,30), per_channel=False, p=0.4),
        A.ISONoise(intensity=(0.05,0.5), p=0.5),
        ], p=1),
    #-----Compression
    A.ImageCompression(quality_lower=30, quality_upper=95, p=1)
], p=1)

lowres_second = A.Compose([
    #-----Blur
    A.OneOf([
        A.GaussianBlur(blur_limit=(7,21), sigma_limit=(0.2,3), p=0.7),
        A.AdvancedBlur(blur_limit=(7,21), sigmaX_limit=(0.2,3), sigmaY_limit=(0.2,3), beta_limit=(0.5,4), p=0.15),
        A.RingingOvershoot(blur_limit=(7,21), p=0.1)
        ], p=0.2),
    #-----Downscale     
    A.OneOf([
        # A.Downscale(scale_min=1/LOW_SCALE, scale_max=1/LOW_SCALE, interpolation=cv2.INTER_NEAREST, p=1),
        A.Downscale(scale_min=1/LOW_SCALE, scale_max=1/LOW_SCALE, interpolation=cv2.INTER_LINEAR, p=1),
        A.Downscale(scale_min=1/LOW_SCALE, scale_max=1/LOW_SCALE, interpolation=cv2.INTER_AREA, p=1),
        A.Downscale(scale_min=1/LOW_SCALE, scale_max=1/LOW_SCALE, interpolation=cv2.INTER_CUBIC, p=1),
        ], p=1),
    #-----Noise
    A.OneOf([
        A.GaussNoise(var_limit=(1,25), p=0.5),
        A.GaussNoise(var_limit=(1,30), per_channel=False, p=0.4),
        A.ISONoise(intensity=(0.05,0.1), p=0.5)
        ], p=1),
    #-----Compression
    A.OneOf([
        A.Compose([
            A.ImageCompression(quality_lower=30, quality_upper=95, p=1),
            A.RingingOvershoot(blur_limit=(7,21), p=0.8)], p=1),
        A.Compose([
            A.RingingOvershoot(blur_limit=(7,21), p=0.8),
            A.ImageCompression(quality_lower=30, quality_upper=95, p=1)], p=1)
        ], p=1)
], p=1)

highres_first = A.Compose([
    A.Sharpen(alpha=(0.5, 0.5), lightness=(1, 1), p=1),
    A.MedianBlur(blur_limit=(3,3), p=0.7)
], p=1)


#####--------------------------Simple
lowres = A.Compose([
    A.Downscale(scale_min=1/LOW_SCALE, scale_max=1/LOW_SCALE, interpolation=1, p=1),
], p=1)

highres = A.Compose([
    A.Sharpen(alpha=(0.6, 0.6), lightness=(1, 1), p=1),
], p=1)


class ImageTransform:
    def __init__(self, path):
        self.path = path
        self.labels = os.listdir(path)
        # self.data = []
        # self.dict_classes = {key:idx for idx,key in enumerate(self.class_names)}
        self.data_labels = {label:[] for label in self.labels}
        self.len_labels = {label:0 for label in self.labels}

        # for idx, label in enumerate(self.labels):
        for label in self.labels:
            files = os.listdir(os.path.join(path, label))
            # self.data += list(zip(files, [idx] * len(files)))
            self.data_labels[label] += files
            self.len_labels[label] += len(files)

    def data_paired(self):
        for label in self.labels[-2:-1]:
            new_root_and_dir_low = create_folder(path="", sub_dir="dataset_paired\\lowResolution_x4_complex", folder_name="")
            new_root_and_dir_high = create_folder(path="", sub_dir="dataset_paired\\highResolution", folder_name="")
            root_and_dir = os.path.join(self.path, label)
            files = os.listdir(root_and_dir)
            for file in tqdm(files):
                image = cv2.imread(os.path.join(root_and_dir, file))
                try:
                    low_res_image = lowres_first(image=image)['image']
                    low_res_image = lowres_second(image=low_res_image)['image']
                    new_img_filename_low = os.path.join(new_root_and_dir_low, file)
                    new_img_filename_high = os.path.join(new_root_and_dir_high, file)
                    # cv2.imwrite(new_img_filename_low, low_res_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                    # cv2.imwrite(new_img_filename_high, high_res_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                except Exception:
                    continue

def test():
    dataset_name = "dataset_noArtifact"
    newDataset_name = "dataset_paired"
    root_rawDataset = f"{os.getcwd()}\\{dataset_name}"
    root_newDataset = f"{os.getcwd()}\\{newDataset_name}"

    low_dir = create_folder(path="", sub_dir=newDataset_name, folder_name="lowResolution_x4_complex")
    # high_dir = create_folder(path="", sub_dir=newDataset_name, folder_name="highResolution")

    dataset = ImageTransform(path=root_rawDataset)
    dataset_paired = dataset.data_paired()

if __name__ == "__main__":
    test()



