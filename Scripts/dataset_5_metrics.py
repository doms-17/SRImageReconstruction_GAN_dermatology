import os
import cv2
from random import random
from skimage import measure


# class Metrics:
#     def __init__(self):
#         pass

#     def psnr(self, img_gt, img_test):
#         result = measure.compare_psnr(img_gt, img_test)
#         return result

#     def ssim(self, img_gt, img_test):
#         img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
#         img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
#         result = measure.compare_ssim(img_gt, img_test, full=True)
#         return result

# --------- metrics
path = "D:\\DOMI\\University\\Magistrale\\Tesi\\Pipeline_coding\\dataset_noArtifact\\MEL"

images = os.listdir(path)
rnd = random.randint(0, len(images))
for image_name in images[rnd:rnd+1]:
    image_name = random.choice(images)
print(image_name)
# image_name = "ISIC_0063845.png"

# image = cv2.imread(os.path.join(path, image_name))
