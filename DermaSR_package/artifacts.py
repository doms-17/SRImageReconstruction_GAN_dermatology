import cv2
import numpy as np
import skimage
from skimage.morphology import remove_small_objects, remove_small_holes, area_closing



class Artifacts:
    """ Artifacts """
    def __init__(self, image):
        self.image = image
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]

    def __repr__(self) -> str:
        return "<class 'Artifacts'>" 

    def __str__(self) -> str:
        return f"Image: {self.image}, Height: {self.height}, Width: {self.width}"

    
    def center_cropping(self, dim_h, dim_w):
        """Crop image of selected height and width starting from center
        
            Params:
            -- threshold_image:                indicates how many pixels we are going to “skip” in both the (x, y) direction

            Returns:
            -- 
                    
        """
        center = self.height/2, self.width/2
        x = center[1] - dim_w/2
        y = center[0] - dim_h/2
        cropped_image = self.image[int(y):int(y + dim_h), int(x):int(x + dim_w)]
        return cropped_image


    def thresholding(self, denoise_kernel, areaClose_kernel):
        """Applying Denoising+Thresholding and doing Area Closing
        
            Params:
            -- threshold_image:                indicates how many pixels we are going to “skip” in both the (x, y) direction

            Returns:
            -- 
            
        """
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (denoise_kernel, denoise_kernel), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        final_image = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones(
            (areaClose_kernel, areaClose_kernel), np.uint8))
        return final_image


    def cropping_to_max_cont(self, threshold_image):
        """Cropping to the maximum contour found

            Params:
            -- threshold_image:                indicates how many pixels we are going to “skip” in both the (x, y) direction

            Returns:
            -- 
            
        """
        contours, _ = cv2.findContours(
            threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None
        contour = max(contours, key=cv2.contourArea)
        # computes the bounding box for the contour
        (x, y, w, h) = cv2.boundingRect(contour)
        coordinates = (x, y, w, h)
        cropped_image = self.image[y:y + h, x:x + w]
        return cropped_image, coordinates, len(contours)


    def sliding_window(self, stepSize, windowSize):
        """Slide a window across the image and yield it

            Params:
            -- stepSize:                indicates how many pixels we are going to “skip” in both the (x, y) direction
            -- windowSize (winW, winH): defines the width and height (in terms of pixels) of the window we are going to extract

            Returns:
            -- The sliced image
            """
        for y in range(0, self.height, stepSize):
            for x in range(0, self.width, stepSize):
                # yield the current window
                yield (x, y, self.image[y:y + windowSize[1], x:x + windowSize[0]])


    def sliding_resize(self, height, width, dimToRes):
        """Slide a window across the image and yield it

            Params:
            -- stepSize:                indicates how many pixels we are going to “skip” in both the (x, y) direction
            -- windowSize (winW, winH): defines the width and height (in terms of pixels) of the window we are going to extract

            Returns:
            -- The sliced image

            """
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
            self.image, (width_resized, height_resized), interpolation=cv2.INTER_AREA)
        img_a = resized_image[:dimToRes, :dimToRes]
        img_b = Artifacts.center_cropping(resized_image, dimToRes, dimToRes)
        img_c = resized_image[-dimToRes:, -dimToRes:]
        return img_a, img_b, img_c


    def hair_removal(self):
        """Removing hair from skin images
            Params:
            -- stepSize:                indicates how many pixels we are going to “skip” in both the (x, y) direction
            -- windowSize (winW, winH): defines the width and height (in terms of pixels) of the window we are going to extract

            Returns:
            -- The sliced image
                    
        """
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
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
        inpainted_image = cv2.inpaint(self.image, th, 1, cv2.INPAINT_TELEA)
        return inpainted_image