import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.morphology import remove_small_objects, remove_small_holes, area_closing
from tqdm import tqdm

HIGH_RES = 1024
BORDER_THICK = 5   # thickness to check on borders when checking for Dark Corner 
DARK_THRESH = 15   # threshold to classify a border as near-black or not and so to state if there is Dark Corner
BORDER_TO_PAD = 200    # thickness of borders to pad to the image

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

def create_folder(path, sub_dir, folder_name):
    path = os.getcwd()
    dir_name = os.path.join(path, sub_dir, folder_name)
    os.makedirs(dir_name, exist_ok=True)
    return dir_name

def thresholding(image, denoise_kernel, areaClose_kernel):
    """Applying Denoising+Thresholding and doing Area Closing"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(denoise_kernel,denoise_kernel),0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((areaClose_kernel,areaClose_kernel),np.uint8))
    # _, th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)  # Adaptive Thresholding
    # th = cv2.erode(th, kernel, iterations=1)
    # th = cv2.dilate(th, kernel, iterations=1)
    return th

def cropping_to_max_cont(image, th):
    """Cropping to the maximum contour found"""
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    contour = max(contours, key = cv2.contourArea)
    (x,y,w,h) = cv2.boundingRect(contour)   # computes the bounding box for the contour
    coordinates = (x,y,w,h)
    cropped_image = image[y:y + h, x:x + w]
    return cropped_image, coordinates, len(contours)

def center_cropping(image, h, w):
    """Crop image of selected height and width starting from center"""
    center = image.shape[0]/2, image.shape[1]/2
    x = center[1] - w/2
    y = center[0] - h/2
    cropped_image = image[int(y):int(y + h), int(x):int(x + w)]
    return cropped_image

def border_padding(image, image_to_pad, coord, thick_to_add):
    """Adding a border frame of selected thickness to the image"""
    x,y,w,h = coord
    max_dim = h if h>w else w
    padded_image = cv2.vconcat([image[y-thick_to_add:y, x:x + max_dim], image_to_pad])
    padded_image = cv2.vconcat([padded_image, image[y + max_dim:y + max_dim + thick_to_add, x:x + max_dim]])
    padded_image = cv2.hconcat([image[y-thick_to_add:y + max_dim + thick_to_add, x - thick_to_add:x], padded_image])
    padded_image = cv2.hconcat([padded_image, image[y - thick_to_add:y + max_dim + thick_to_add, x + max_dim:x + max_dim + thick_to_add]])
    return padded_image

def sliding_window(image, stepSize, windowSize):
    """slide a window across the image"""
    for y in range(0, image.shape[0], stepSize):
	    for x in range(0, image.shape[1], stepSize):
			# yield the current window
		    yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


class ImageArtifact:
    def __init__(self, path):
        self.path = path
        self.labels = os.listdir(path)
        # self.data = []
        # self.dict_classes = {key:idx for idx,key in enumerate(self.class_names)}
        self.data_labels = {label:[] for label in self.labels}
        self.len_labels = {label:0 for label in self.labels}
        self.len_labels_post = {label:0 for label in self.labels}

        # for idx, label in enumerate(self.labels):
        for label in self.labels:
            files = os.listdir(os.path.join(path, label))
            # self.data += list(zip(files, [idx] * len(files)))
            self.data_labels[label] += files
            self.len_labels[label] += len(files)

    def __len__(self):
        return len(self.data)

    def handle_artifacts(self):
        """Handle artifact:
        - first check if image has Dark Corner artifact, if not check if it is 1024x1024, then manage the rectangular image"""
        for label in self.labels:
            new_root_and_dir = create_folder(path="", sub_dir="dataset_noArtifact2", folder_name=label)
            root_and_dir = os.path.join(self.path, label)
            files = os.listdir(root_and_dir)
            for file in tqdm(files):
                image = cv2.imread(os.path.join(root_and_dir, file))
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                height, width, _ = image.shape
                # if (height >= HIGH_RES and width >= HIGH_RES) \
                # or (height == width and 800<height<HIGH_RES and 800<width<HIGH_RES) \
                # or (height == HIGH_RES and 700<width<HIGH_RES) or (width == HIGH_RES and 700<height<HIGH_RES):
                sx_border, dx_border = np.mean(gray[:,0:BORDER_THICK]), np.mean(gray[:,-BORDER_THICK:])
                up_border, down_border = np.mean(gray[0:BORDER_THICK,:]), np.mean(gray[-BORDER_THICK:,:])
                try:
                    # if two borders of the grayscale image have near-black pixels, it is "probably" Dark Corner
                    if ( (sx_border and dx_border) or (up_border and down_border) or \
                        (sx_border and up_border) or (dx_border and down_border) or \
                        (sx_border and down_border) or (dx_border and up_border) ) < DARK_THRESH:
                        th = thresholding(image, denoise_kernel=5, areaClose_kernel=50)
                        results = cropping_to_max_cont(image, th)
                        if results is not None:
                            final_image, _, _ = results
                            final_image = center_cropping(final_image, min(height,width), min(height,width))
                            height, width, _ = final_image.shape
                            if height > HIGH_RES and width > HIGH_RES:
                                final_image = cv2.resize(final_image, (HIGH_RES,HIGH_RES), interpolation=cv2.INTER_AREA) # shrinking
                            elif height < HIGH_RES and width < HIGH_RES:
                                final_image = cv2.resize(final_image, (HIGH_RES,HIGH_RES), interpolation=cv2.INTER_CUBIC) # enlarge
                    # if image is not Dark Corner and it has square dimension
                    elif height == width:  
                        if height > HIGH_RES:
                            final_image = cv2.resize(final_image, (HIGH_RES,HIGH_RES), interpolation=cv2.INTER_AREA)
                        # elif height < HIGH_RES:
                        #     final_image = cv2.resize(final_image, (HIGH_RES,HIGH_RES), interpolation=cv2.INTER_CUBIC)
                        else:   # height == HIGH_RES:
                            final_image = image
                    # if image is neither Dark Corner or squared, then it is rectangular
                    else:
                        center_image = center_cropping(image, min(height,width), min(height,width))
                        height, width, _ = center_image.shape
                        if height > HIGH_RES:
                            th = thresholding(center_image, denoise_kernel=5, areaClose_kernel=50)
                            th = cv2.bitwise_not(th) # invert the thresholding mask
                            cropped_image, coordinates, num_conts = cropping_to_max_cont(center_image, th)
                            if num_conts < 10:    
                                x,y,w,h = coordinates
                                if w > h:
                                    square_image = cv2.vconcat([cropped_image, center_image[y+h:y+w, x:x+w]])
                                    max_dim = w
                                elif h > w:
                                    square_image = cv2.hconcat([cropped_image, center_image[y:y+h, x+w:x+h]])
                                    max_dim = h
                                elif h == w:
                                    square_image = cropped_image
                                    max_dim = w

                                if y-BORDER_TO_PAD>=0 and x-BORDER_TO_PAD>=0 and y+max_dim+BORDER_TO_PAD<height and x+max_dim+BORDER_TO_PAD<width:
                                    final_image = border_padding(center_image, square_image, coordinates, BORDER_TO_PAD)
                                else:
                                    final_image = square_image
                            else:
                                final_image = center_image
                        else:
                            final_image = center_image

                        final_image = cv2.resize(final_image, (HIGH_RES,HIGH_RES), interpolation=cv2.INTER_AREA)

                    # Saving image
                    new_img_filename = os.path.join(new_root_and_dir, file) 
                    cv2.imwrite(new_img_filename, final_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                    
                except Exception:
                    continue

def test():
    dataset_name = "Original_orsize"
    newDataset_name = "dataset_noArtifact_ESRGAN"
    root_rawDataset = f"{os.getcwd()}\\{dataset_name}"
    root_cropDataset = f"{os.getcwd()}\\{newDataset_name}"

    new_dir = create_folder(path="", sub_dir="", folder_name=newDataset_name)

    dataset = ImageArtifact(path=root_rawDataset)
    dataset_noArtifacts = dataset.handle_artifacts()

if __name__ == "__main__":
    test()