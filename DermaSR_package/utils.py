import os
from matplotlib import pyplot as plt


def plot_img(img):
    plt.figure(figsize=(20, 10))
    plt.imshow(img)
    plt.show()


def plot_images(img1, img2):
    plt.figure(figsize=(20, 10))
    ax1 = plt.subplot(211)
    ax1.imshow(img1)
    ax2 = plt.subplot(212)
    ax2.imshow(img2)
    plt.show()


def create_folder(dir_name):
    os.makedirs(dir_name, exist_ok=True)
    return dir_name
