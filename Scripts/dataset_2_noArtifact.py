import os
import json
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.morphology import remove_small_objects, remove_small_holes, area_closing
from tqdm import tqdm

HIGH_RES = 1024
BORDER_THICK = 5        # thickness to check on borders when checking for Dark Corner 
DARK_THRESH = 15        # threshold to classify a border as near-black or not and so to state if there is Dark Corner
THRESH_RATIO = 1000     # threshold to activate slidinw_window or not

COMPRESSION = 6

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
    blur = cv2.GaussianBlur(gray,(denoise_kernel,denoise_kernel),0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((areaClose_kernel,areaClose_kernel),np.uint8))
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

def sliding_window(image, stepSize, windowSize):
    """Slide a window across the image

        Params:
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
    ratio_or = max(height,width)/min(height,width)
    if height < width:
        height_resized = dimToRes
        width_resized = int(width*(height_resized/height))
    else:
        width_resized = dimToRes
        height_resized = int(height*(width_resized/width))
    # ratio_new = max(height_resized,width_resized)/min(height_resized,width_resized)
    # if not (ratio_or-3 <ratio_new< ratio_or+3):
    #     return None
    resized_image = cv2.resize(image, (width_resized,height_resized), interpolation=cv2.INTER_AREA)
    img_a = resized_image[:dimToRes,:dimToRes]
    img_b = center_cropping(resized_image, dimToRes, dimToRes)
    img_c = resized_image[-dimToRes:,-dimToRes:]
    return img_a, img_b, img_c

class ImageArtifact:
    def __init__(self, path):
        self.path = path
        self.labels = os.listdir(path)

        # self.path_json = f"{os.getcwd()}\\file_selected_per_label.json"

        self.files_per_labels = {label:[] for label in self.labels}
        self.files_noArtifact = {label:[] for label in self.labels}

        json_to_dict = json.load(open("D:\\DOMI\\University\\Magistrale\\Tesi\\Pipeline_coding\\files_selected_per_label.json"))
        self.files_manually_selected = json_to_dict["files_per_label"]

        self.len_per_labels = {label:0 for label in self.labels}
        self.len_labels_noArtifact = {label:0 for label in self.labels}

        # for idx, label in enumerate(self.labels):
        for label in self.labels:
            files = os.listdir(os.path.join(path, label))
            self.files_per_labels[label] += files
            self.len_per_labels[label] += len(files)
            

    def handle_artifacts(self):
        """Handle artifact:
        - 1) select images based on size:
            - heigth&width >= 1024 ---> Ok
            - 1024x720 or 720x1024 ---> No
            - 850x850 ---> No
            - 720x720 ---> No
        - 2) check if image has Dark Corner artifact, if not check if it is 1024x1024, otherwise manage the rectangular image
        """
        for label in self.labels:
            new_root_and_dir = create_folder(path="", sub_dir="dataset_noArtifact_sliding", folder_name=label) # create folder per each label
            root_and_dir = os.path.join(self.path, label) # retrieve path of preDataset

            for file in tqdm(self.files_per_labels[label]):
                if file in self.files_manually_selected[label]:
                    image = cv2.imread(os.path.join(root_and_dir, file))
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    height, width, _ = image.shape
                    
                    # Check if image is >= 1024
                    if (height >= HIGH_RES and width >= HIGH_RES):          
                    # or (height == width and 800<height<HIGH_RES and 800<width<HIGH_RES) \
                    # or (height == HIGH_RES and 700<width<HIGH_RES) or (width == HIGH_RES and 700<height<HIGH_RES):
                        sx_border, dx_border = np.mean(gray[:,0:BORDER_THICK]), np.mean(gray[:,-BORDER_THICK:])
                        up_border, down_border = np.mean(gray[0:BORDER_THICK,:]), np.mean(gray[-BORDER_THICK:,:])
                        
                        try:
                            # If two borders of the grayscale image have near-black pixels, it is "probably" Dark Corner
                            if ( (sx_border and dx_border) or (up_border and down_border)  \
                                or (sx_border and up_border) or (dx_border and down_border)  \
                                or (sx_border and down_border) or (dx_border and up_border) ) < DARK_THRESH:
                                th = thresholding(image, denoise_kernel=5, areaClose_kernel=50)
                                results = cropping_to_max_cont(image, th)
                                if results is not None:
                                    noDarkCorner_image, _, _ = results
                                    height, width, _ = noDarkCorner_image.shape
                                    center_image = center_cropping(noDarkCorner_image, min(height,width), min(height,width))
                                    # height, width, _ = final_image.shape
                                    if center_image.shape[0] > HIGH_RES and center_image.shape[1] > HIGH_RES:
                                        final_image = cv2.resize(center_image, (HIGH_RES,HIGH_RES), interpolation=cv2.INTER_AREA) # shrinking
                                    elif center_image.shape[0] < HIGH_RES and center_image.shape[1] < HIGH_RES:
                                        final_image = cv2.resize(center_image, (HIGH_RES,HIGH_RES), interpolation=cv2.INTER_CUBIC) # enlarge
                                    elif center_image.shape[0] == HIGH_RES and center_image.shape[1] == HIGH_RES:
                                        final_image = center_image
                                    # Saving image
                                    if final_image.shape[0] == HIGH_RES and final_image.shape[1] == HIGH_RES:
                                        new_img_filename = os.path.join(new_root_and_dir, file)
                                        cv2.imwrite(new_img_filename, final_image, [cv2.IMWRITE_PNG_COMPRESSION, COMPRESSION])
                                        
                            # If image is not Dark Corner and it has square dimension
                            elif height == width:  
                                if height > HIGH_RES:
                                    final_image = cv2.resize(image, (HIGH_RES,HIGH_RES), interpolation=cv2.INTER_AREA)
                                elif height < HIGH_RES:
                                    final_image = cv2.resize(image, (HIGH_RES,HIGH_RES), interpolation=cv2.INTER_CUBIC)
                                elif height == HIGH_RES:
                                    final_image = image
                                # Saving image
                                if final_image.shape[0] == HIGH_RES and final_image.shape[1] == HIGH_RES:
                                    new_img_filename = os.path.join(new_root_and_dir, file)
                                    cv2.imwrite(new_img_filename, final_image, [cv2.IMWRITE_PNG_COMPRESSION, COMPRESSION])

                            # If image is neither Dark Corner or squared, then it is rectangular
                            else:        
                                if max(height,width)-min(height,width) >= THRESH_RATIO:
                                    # results = sliding_resize(image, height, width, HIGH_RES)
                                    # if results is not None:
                                        # (image_a, image_b, image_c) = results
                                    image_a, image_b, image_c = sliding_resize(image, height, width, HIGH_RES)
                                    id_image = file.split('.')[0]
                                    file_reza = f'{id_image}_slia.png'
                                    file_rezb = f'{id_image}_slib.png'
                                    file_rezc = f'{id_image}_slic.png'

                                    new_img_filename = os.path.join(new_root_and_dir, file_reza) 
                                    cv2.imwrite(new_img_filename, image_a, [cv2.IMWRITE_PNG_COMPRESSION, COMPRESSION])
                                    new_img_filename = os.path.join(new_root_and_dir, file_rezb) 
                                    cv2.imwrite(new_img_filename, image_b, [cv2.IMWRITE_PNG_COMPRESSION, COMPRESSION])
                                    new_img_filename = os.path.join(new_root_and_dir, file_rezc) 
                                    cv2.imwrite(new_img_filename, image_c, [cv2.IMWRITE_PNG_COMPRESSION, COMPRESSION])
                                else:
                                    _, final_image, _ = sliding_resize(image, height, width, HIGH_RES)
                                    # Saving image
                                    if final_image.shape[0] == HIGH_RES and final_image.shape[1] == HIGH_RES:
                                        new_img_filename = os.path.join(new_root_and_dir, file) 
                                        cv2.imwrite(new_img_filename, final_image, [cv2.IMWRITE_PNG_COMPRESSION, COMPRESSION])
                                            
                        except Exception:
                            continue

def test():
    dataset_name = "Original_orsize"
    newDataset_name = "dataset_noArtifact_sliding"

    root_rawDataset = f"{os.getcwd()}\\{dataset_name}"
    root_noArtifactDataset = f"{os.getcwd()}\\{newDataset_name}"

    create_folder(path="", sub_dir="", folder_name=newDataset_name)

    dataset = ImageArtifact(path=root_rawDataset)
    dataset.handle_artifacts()

if __name__ == "__main__":
    test()