import os
import json
import cv2
import numpy as np
import skimage
from matplotlib import pyplot as plt
from skimage.morphology import remove_small_objects, remove_small_holes, area_closing
from tqdm import tqdm

HIGH_RES = 1024
BORDER_THICK = 30       # thickness to check on borders when checking for Dark Corner

DARK_THRESH = 10    # threshold to classify a border as near-black or not

THRESH_RATIO = 1000     # threshold to activate slidinw_window or not

COMPRESSION = 6


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


def hair_removal(image):
    """Removing hair from skin images"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Perform the blackHat filtering on the grayscale image to find the hair countours
    blackhat = cv2.morphologyEx(
        gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_CROSS, (17, 17)))
    # plot_images(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), blackhat)
    # intensify the hair countours in preparation for the inpainting algorithm
    _, th = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    th = skimage.morphology.remove_small_objects(
        th.astype(bool), 700, connectivity=3).astype(np.uint8)
    th = cv2.dilate(th, cv2.getStructuringElement(
        cv2.MORPH_CROSS, (3, 3)), iterations=1)
    # plot_images(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), th)
    # inpaint the original image depending on the mask
    inpaint = cv2.inpaint(image, th, 1, cv2.INPAINT_TELEA)
    return inpaint


def center_cropping(image, h, w):
    """Crop image of selected height and width starting from center"""
    center = image.shape[0]/2, image.shape[1]/2
    x = center[1] - w/2
    y = center[0] - h/2
    cropped_image = image[int(y):int(y + h), int(x):int(x + w)]
    return cropped_image


def thresholding(image, denoise_kernel, areaClose_kernel):
    """Applying Denoising+Thresholding and doing Area Closing"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (denoise_kernel, denoise_kernel), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones(
        (areaClose_kernel, areaClose_kernel), np.uint8))
    return th


def cropping_to_max_cont(image, th):
    """Cropping to the maximum contour found"""
    contours, _ = cv2.findContours(
        th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    contour = max(contours, key=cv2.contourArea)
    # computes the bounding box for the contour
    (x, y, w, h) = cv2.boundingRect(contour)
    coordinates = (x, y, w, h)
    cropped_image = image[y:y + h, x:x + w]
    return cropped_image, coordinates, len(contours)


def sliding_window(image, stepSize, windowSize):
    """Slide a window across the image and yield it

        Params:
        -- image:                   image input
        -- stepSize:                indicates how many pixels we are going to “skip” in both the (x, y) direction
        -- windowSize (winW, winH): defines the width and height (in terms of pixels) of the window we are going to extract

        Returns:
        -- The sliced image
        """
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def sliding_resize(image, height, width, dimToRes):
    ratio_or = max(height, width)/min(height, width)
    if height < width:
        height_resized = dimToRes
        width_resized = int(width*(height_resized/height))
    else:
        width_resized = dimToRes
        height_resized = int(height*(width_resized/width))
    # ratio_new = max(height_resized,width_resized)/min(height_resized,width_resized)
    # if not (ratio_or-3 <ratio_new< ratio_or+3):
    #     return None
    resized_image = cv2.resize(
        image, (width_resized, height_resized), interpolation=cv2.INTER_AREA)
    img_a = resized_image[:dimToRes, :dimToRes]
    img_b = center_cropping(resized_image, dimToRes, dimToRes)
    img_c = resized_image[-dimToRes:, -dimToRes:]
    return img_a, img_b, img_c


def handle_artifacts(image_filename):
    """Handle artifact:
    - 1) select images based on size:
        - heigth&width >= 1024 ---> Ok
        - 1024x720 or 720x1024 ---> No
        - 850x850 ---> No
        - 720x720 ---> No
    - 2) check if image has Dark Corner artifact, if not check if it is 1024x1024, otherwise manage the rectangular image
    """

    # if file in self.files_manually_selected[label]:
    image = cv2.imread(image_filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width, _ = image.shape
    th = thresholding(
        image, denoise_kernel=5, areaClose_kernel=50)

    # Check if image is >= 1024
    # if (height >= 0 and width >= 0):
    # or (height == width and 800<height<HIGH_RES and 800<width<HIGH_RES) \
    # or (height == HIGH_RES and 700<width<HIGH_RES) or (width == HIGH_RES and 700<height<HIGH_RES):
    sx_border = np.mean(th[:, 0:BORDER_THICK])
    dx_border = np.mean(th[:, -BORDER_THICK:])
    up_border = np.mean(th[0:BORDER_THICK, :])
    down_border = np.mean(th[-BORDER_THICK:, :])

    # If two borders of the grayscale image have near-black pixels, it is "probably" Dark Corner
    if ((sx_border and dx_border) or (up_border and down_border)
        or (sx_border and up_border) or (dx_border and down_border)
            or (sx_border and down_border) or (dx_border and up_border)) < DARK_THRESH:

        results = cropping_to_max_cont(image, th)
        if results is not None:
            noDarkCorner_image, _, _ = results
            height, width, _ = noDarkCorner_image.shape
            center_image = center_cropping(
                noDarkCorner_image, min(height, width), min(height, width))
            # height, width, _ = final_image.shape
            if center_image.shape[0] > HIGH_RES and center_image.shape[1] > HIGH_RES:
                final_image = cv2.resize(
                    center_image, (HIGH_RES, HIGH_RES), interpolation=cv2.INTER_AREA)  # shrinking
            elif center_image.shape[0] < HIGH_RES and center_image.shape[1] < HIGH_RES:
                final_image = cv2.resize(
                    center_image, (HIGH_RES, HIGH_RES), interpolation=cv2.INTER_CUBIC)  # enlarge
            elif center_image.shape[0] == HIGH_RES and center_image.shape[1] == HIGH_RES:
                final_image = center_image

        return final_image

    # If image is not Dark Corner and it has square dimension
    elif height == width:

        if (height == width and height > HIGH_RES):
            final_image = cv2.resize(
                image, (HIGH_RES, HIGH_RES), interpolation=cv2.INTER_AREA)
        elif (height == width and height < HIGH_RES):
            final_image = cv2.resize(
                image, (HIGH_RES, HIGH_RES), interpolation=cv2.INTER_CUBIC)
        elif (height == width and height == HIGH_RES):
            final_image = image

        return final_image

    # If image is neither Dark Corner or squared, then it is rectangular
    else:
        if max(height, width)-min(height, width) >= THRESH_RATIO:

            image_a, image_b, image_c = sliding_resize(
                image, height, width, HIGH_RES)
            id_image = image_filename.split('\\')[-1].split('.')[0]
            file_a = f'{id_image}_a.png'
            file_b = f'{id_image}_b.png'
            file_c = f'{id_image}_c.png'

            return (file_a, file_b, file_c), (image_a, image_b, image_c)

        else:

            if (height >= HIGH_RES and width >= HIGH_RES):
                _, final_image, _ = sliding_resize(
                    image, height, width, HIGH_RES)
            else:
                center_image = center_cropping(
                    image, min(height, width), min(height, width))
                final_image = cv2.resize(
                    center_image, (HIGH_RES, HIGH_RES), interpolation=cv2.INTER_CUBIC)  # enlarge

            return final_image


def test():
    root_test_sets = "D:\\DOMI\\University\\Magistrale\\Tesi\\Pipeline_coding\\Test Set\\to_test_lab"
    new_root_test_sets = "D:\\DOMI\\University\\Magistrale\\Tesi\\Pipeline_coding\\Test Set\\to_test_lab_noArtifacts"

    # root_test_sets = "D:\\DOMI\\University\\Magistrale\\Tesi\\Pipeline_coding\\Test Set\\to_test_Atlas\\images"
    # new_root_test_sets = "D:\\DOMI\\University\\Magistrale\\Tesi\\Pipeline_coding\\Test Set\\to_test_Atlas\\images_noArtifacts"

    test_sets = os.listdir(root_test_sets)

    for test_set in test_sets:
        new_root_test_set = f"{os.path.join(new_root_test_sets, test_set)}_new"
        new_root_test_set = f"{os.path.join(new_root_test_sets, test_set)}_new"
        create_folder(new_root_test_set)
        files = os.listdir(os.path.join(root_test_sets, test_set))
        for file in tqdm(files):
            complete_filename = os.path.join(root_test_sets, test_set, file)
            try:
                no_artifact_image = handle_artifacts(complete_filename)
                if len(no_artifact_image) == 2:
                    filename_a, filename_b, filename_c = no_artifact_image[0]
                    no_artifact_image_a, no_artifact_image_b, no_artifact_image_c = no_artifact_image[
                        1]
                    # Saving image
                    if ((no_artifact_image_a.shape[0] == HIGH_RES and no_artifact_image_a.shape[1] == HIGH_RES)
                        and (no_artifact_image_b.shape[0] == HIGH_RES and no_artifact_image_b.shape[1] == HIGH_RES)
                            and (no_artifact_image_c.shape[0] == HIGH_RES and no_artifact_image_c.shape[1] == HIGH_RES)):
                        new_img_filename = os.path.join(
                            new_root_test_set, filename_a)
                        cv2.imwrite(new_img_filename, no_artifact_image, [
                                    cv2.IMWRITE_PNG_COMPRESSION, COMPRESSION])
                        new_img_filename = os.path.join(
                            new_root_test_set, filename_b)
                        cv2.imwrite(new_img_filename, no_artifact_image, [
                                    cv2.IMWRITE_PNG_COMPRESSION, COMPRESSION])
                        new_img_filename = os.path.join(
                            new_root_test_set, filename_c)
                        cv2.imwrite(new_img_filename, no_artifact_image, [
                                    cv2.IMWRITE_PNG_COMPRESSION, COMPRESSION])
                else:
                    # Saving image
                    if no_artifact_image.shape[0] == HIGH_RES and no_artifact_image.shape[1] == HIGH_RES:
                        new_img_filename = os.path.join(
                            new_root_test_set, file)
                        cv2.imwrite(new_img_filename, no_artifact_image, [
                                    cv2.IMWRITE_PNG_COMPRESSION, COMPRESSION])
            except Exception:
                continue


if __name__ == "__main__":
    test()
