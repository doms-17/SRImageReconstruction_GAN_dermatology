import albumentations as A
import cv2



class Paired:
    """ Paired """
    def __init__(self, image):
        self.image = image
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]

    def __repr__(self) -> str:
        return "<class 'Paired'>" 

    def __str__(self) -> str:
        return f"Image: {self.image}, Height: {self.height}, Width: {self.width}"


    def degradation_1order(self, scale):
        """Removing hair from skin images
            Params:
            -- stepSize:                indicates how many pixels we are going to “skip” in both the (x, y) direction
            -- windowSize (winW, winH): defines the width and height (in terms of pixels) of the window we are going to extract

            Returns:
            -- The sliced image

        """
        lowres = A.Compose([
            ########## 1ST ORDER ##########
            # -----Blur
            A.OneOf([
                A.GaussianBlur(blur_limit=(7, 21),
                                sigma_limit=(0.2, 3), p=0.7),
                A.AdvancedBlur(blur_limit=(7, 21), sigmaX_limit=(
                    0.2, 3), sigmaY_limit=(0.2, 3), beta_limit=(0.5, 4), p=0.15),
                A.RingingOvershoot(blur_limit=(7, 21), p=0.15),
            ], p=1),
            # -----Downscale
            A.OneOf([
                # A.Downscale(scale_min=1/LOW_SCALE, scale_max=1/LOW_SCALE, interpolation=cv2.INTER_NEAREST, p=1),
                A.Downscale(scale_min=1/scale, scale_max=1/scale,
                            interpolation=cv2.INTER_LINEAR, p=1),
                A.Downscale(scale_min=1/scale, scale_max=1/scale,
                            interpolation=cv2.INTER_AREA, p=1),
                A.Downscale(scale_min=1/scale, scale_max=1/scale,
                            interpolation=cv2.INTER_CUBIC, p=1),
            ], p=1),
            # -----Noise
            A.OneOf([
                A.GaussNoise(var_limit=(1, 30), p=0.4),
                A.GaussNoise(var_limit=(1, 30), per_channel=False, p=0.2),
                A.ISONoise(intensity=(0.05, 0.5), p=0.4),
            ], p=1),
            # -----Compression
            A.ImageCompression(quality_lower=30, quality_upper=95, p=1),
        ],p=1)
        return lowres(image=self.image)['image']
        

    def degradation_norder(self, scale):
        """Removing hair from skin images
            Params:
            -- stepSize:                indicates how many pixels we are going to “skip” in both the (x, y) direction
            -- windowSize (winW, winH): defines the width and height (in terms of pixels) of the window we are going to extract

            Returns:
            -- The sliced image

        """
        lowres = A.Compose([        
            ########## N ORDER ##########
            # -----Blur
            A.OneOf([
                A.GaussianBlur(blur_limit=(7, 21),
                               sigma_limit=(0.2, 3), p=0.7),
                A.AdvancedBlur(blur_limit=(7, 21), sigmaX_limit=(
                    0.2, 3), sigmaY_limit=(0.2, 3), beta_limit=(0.5, 4), p=0.15),
                A.RingingOvershoot(blur_limit=(7, 21), p=0.15),
            ], p=0.8),
            # -----Downscale
            A.OneOf([
                # A.Downscale(scale_min=1/scale, scale_max=1/scale, interpolation=cv2.INTER_NEAREST, p=1),
                A.Downscale(scale_min=1/scale, scale_max=1/scale,
                            interpolation=cv2.INTER_LINEAR, p=1),
                A.Downscale(scale_min=1/scale, scale_max=1/scale,
                            interpolation=cv2.INTER_AREA, p=1),
                A.Downscale(scale_min=1/scale, scale_max=1/scale,
                            interpolation=cv2.INTER_CUBIC, p=1),
            ], p=1),
            # -----Noise
            A.OneOf([
                A.GaussNoise(var_limit=(1, 25), p=0.4),
                A.GaussNoise(var_limit=(1, 30), per_channel=False, p=0.2),
                A.ISONoise(intensity=(0.05, 0.1), p=0.4),
            ], p=1),
            # -----Compression
            A.OneOf([
                A.Compose([
                    A.ImageCompression(
                        quality_lower=30, quality_upper=95, p=0.8),
                    A.RingingOvershoot(blur_limit=(7, 21), p=0.5),
                ], p=1),
                A.Compose([
                    A.RingingOvershoot(blur_limit=(7, 21), p=0.5),
                    A.ImageCompression(
                        quality_lower=30, quality_upper=95, p=0.8),
                ], p=1),
            ], p=1),
        ], p=1)
        return lowres(image=self.image)['image']


    def sharpness(self):
        """Removing hair from skin images
            Params:
            -- stepSize:                indicates how many pixels we are going to “skip” in both the (x, y) direction
            -- windowSize (winW, winH): defines the width and height (in terms of pixels) of the window we are going to extract

            Returns:
            -- The sliced image

        """
        highres_plus = A.Compose([
            A.Sharpen(alpha=(0.5, 0.5), lightness=(1, 1), p=1),
            A.MedianBlur(blur_limit=(3, 3), p=0.5),
        ], p=1)
        return highres_plus(image=self.image)['image']
