import cv2
import numpy as np
import skimage


class Artifacts:
    """ Artifacts solver transforms"""
    def __init__(self, image):
        self.image = image
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]

    def __repr__(self) -> str:
        return "<class 'Artifacts'>" 

    def __str__(self) -> str:
        return f"Image: {self.image}, Height: {self.height}, Width: {self.width}"

    def center_crop(self, crop_height:int, crop_width:int):
        """Crop central image of selected height and width
        
            Params:
            -- crop_height (int): height of the crop.
            -- crop_width (int): width of the crop.

            Returns:
            -- crop_image: cropped image of selected h and w dimension
                    
        """
        center: tuple[int] = self.height/2, self.width/2
        x:int = center[1] - crop_width/2
        y:int = center[0] - crop_height/2
        crop_image = self.image[int(y):int(y + crop_height), int(x):int(x + crop_width)]
        return crop_image


    def get_mask(self, apply_blur:bool=False, blur_kernel:int=0, 
                 apply_morph_close:bool=True, morph_close_kernel:int=50,
                 apply_morph_open:bool=True, morph_open_kernel:int=50):
        """Applying Thresholding to create binary mask from image
        
            Params:
            -- blur_kernel (int): kernel size for blurring the input image (Default=3)
            -- morph_close_kernel (int): kernel size for blurring the input image (Default=50)
            -- morph_open_kernel (int): kernel size for blurring the input image (Default=50)
            -- apply_blur (bool): wheter to apply blur transform (Default=False)
            -- apply_morph_close (bool): wheter to apply close operation transform (Default=True)
            -- apply_morph_open (bool): wheter to apply open operation transform (Default=False)

            Returns:
            -- mask: binary mask of the starting image
            
        """
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        if apply_blur:
            gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if apply_morph_close:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((morph_close_kernel, morph_close_kernel), np.uint8))
        if apply_morph_open:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((morph_open_kernel, morph_open_kernel), np.uint8))
        return mask


    def get_max_contour(self, mask):
        """Get image contours, find the biggest contour and its coordinates, crop the image using coordinates

            Params:
            -- mask: binary mask

            Returns:
            -- crop_image: cropped image
            -- coordinates: coordinates of the biggest contour found
            -- len(contours): number of countours found
            
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None
        contour = max(contours, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(contour)
        crop_image = self.image[y:y + h, x:x + w]
        coordinates: tuple[int] = (x, y, w, h)
        return crop_image, coordinates, len(contours)


    def sliding_window(self, stepSize:int, windowSize:int):
        """Slide a window across the image and yield it

            Params:
            -- stepSize:                indicates how many pixels we are going to “skip” in both the (x, y) direction
            -- windowSize (winW, winH): defines the width and height (in terms of pixels) of the window we are going to extract

            Returns:
            -- Yield sliced images

        """
        for y in range(0, self.height, stepSize):
            for x in range(0, self.width, stepSize):
                # yield the current window
                yield (x, y, self.image[y:y + windowSize[1], x:x + windowSize[0]])


    def rect2square_image_splitting(self, resize_dim:int):
        """Split rectangular images into three square parts

            Params:
            -- resize_dim (int): dimension of the desired output images

            Returns:
            -- img_a: left cropped part of the image
            -- img_b: center cropped part of the image
            -- img_c: right cropped part of the image

        """
        # ratio = max(self.height, self.width)/min(self.height, self.width)
        if self.height < self.width:
            height_resized = resize_dim
            width_resized = int(self.width*(height_resized/self.height))
        else:
            width_resized = resize_dim
            height_resized = int(self.height*(width_resized/self.width))

        resized_image = cv2.resize(self.image, (width_resized, height_resized), interpolation=cv2.INTER_CUBIC)
        img_a = resized_image[:resize_dim, :resize_dim]

        x:int = resized_image.shape[1]/2 - resize_dim/2
        y:int = resized_image.shape[0]/2 - resize_dim/2
        img_b = resized_image[int(y):int(y + resize_dim), int(x):int(x + resize_dim)]

        img_c = resized_image[-resize_dim:, -resize_dim:]
        return img_a, img_b, img_c


    def hair_removal(self):
        """Remove hair from skin images
        
            Params:
            -- none

            Returns:
            -- inpaint_image: inpainted image without hair
                    
        """
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # Perform the blackHat filtering on the grayscale image to find the hair countours
        blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_CROSS, (17, 17)))
        # intensify the hair countours in preparation for the inpainting algorithm
        _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        mask = skimage.morphology.remove_small_objects(mask.astype(bool), 700, connectivity=3).astype(np.uint8)
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=1)
        # inpaint the original image depending on the mask
        inpaint_image = cv2.inpaint(self.image, mask, 1, cv2.INPAINT_TELEA)
        return inpaint_image