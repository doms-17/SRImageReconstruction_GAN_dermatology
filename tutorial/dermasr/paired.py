import albumentations as A
import cv2


class Paired:
    """Pipeline transforms to degradate or enhance an image to its low resolution (LR) or high resolution (HR) counterpart"""

    def __init__(self, image):
        self.image = image
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]

    def __repr__(self) -> str:
        return "<class 'Paired'>"

    def __str__(self) -> str:
        return f"Image: {self.image}, Height: {self.height}, Width: {self.width}"

    def degradation_first_order(self, scale_min:int=4, scale_max:int=4, prob_nearest:float=0):
        """Pipeline transform for 1° order degradation
        
            Params:
            -- scale_min: lower bound on the image scale (Default: 4)
            -- scale_max: upper bound on the image scale (Default: 4)
            -- prob_nearest (float): probability of applying the transform (Default: 0)

            Returns:
            -- lowres: low resolution image

        """
        lowres = A.Compose(
            [
                ########## 1ST ORDER ##########
                # -----Blur
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(7, 21), sigma_limit=(0.2, 3), p=0.7),
                        A.AdvancedBlur(
                            blur_limit=(7, 21),
                            sigmaX_limit=(0.2, 3),
                            sigmaY_limit=(0.2, 3),
                            beta_limit=(0.5, 4),
                            p=0.15,
                        ),
                        A.RingingOvershoot(blur_limit=(7, 21), p=0.15),
                    ],
                    p=1,
                ),
                # -----Downscale
                A.OneOf(
                    [
                        A.Downscale(
                            scale_min=1 / scale_min,
                            scale_max=1 / scale_max,
                            interpolation=cv2.INTER_NEAREST,
                            p=prob_nearest,
                        ),
                        A.Downscale(
                            scale_min=1 / scale_min,
                            scale_max=1 / scale_max,
                            interpolation=cv2.INTER_LINEAR,
                            p=1,
                        ),
                        A.Downscale(
                            scale_min=1 / scale_min,
                            scale_max=1 / scale_max,
                            interpolation=cv2.INTER_AREA,
                            p=1,
                        ),
                        A.Downscale(
                            scale_min=1 / scale_min,
                            scale_max=1 / scale_max,
                            interpolation=cv2.INTER_CUBIC,
                            p=1,
                        ),
                    ],
                    p=1,
                ),
                # -----Noise
                A.OneOf(
                    [
                        A.GaussNoise(var_limit=(1, 30), p=0.4),
                        A.GaussNoise(var_limit=(1, 30), per_channel=False, p=0.2),
                        A.ISONoise(intensity=(0.05, 0.5), p=0.4),
                    ],
                    p=1,
                ),
                # -----Compression
                A.ImageCompression(quality_lower=30, quality_upper=95, p=1),
            ],
            p=1,
        )
        return lowres(image=self.image)["image"]

    def degradation_n_order(self, scale_min:int=4, scale_max:int=4, prob_nearest:float=0):
        """Pipeline transform for N order degradation:
        
            Params:
            -- scale_min: lower bound on the image scale (Default: 4)
            -- scale_max: upper bound on the image scale (Default: 4)
            -- prob_nearest (float): probability of applying the transform (Default: 0)

            Returns:
            -- lowres: low resolution image

        """
        lowres = A.Compose(
            [
                ########## N ORDER ##########
                # -----Blur
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(7, 21), sigma_limit=(0.2, 3), p=0.7),
                        A.AdvancedBlur(
                            blur_limit=(7, 21),
                            sigmaX_limit=(0.2, 3),
                            sigmaY_limit=(0.2, 3),
                            beta_limit=(0.5, 4),
                            p=0.15,
                        ),
                        A.RingingOvershoot(blur_limit=(7, 21), p=0.15),
                    ],
                    p=0.8,
                ),
                # -----Downscale
                A.OneOf(
                    [
                        A.Downscale(
                            scale_min=1 / scale_min,
                            scale_max=1 / scale_max,
                            interpolation=cv2.INTER_NEAREST,
                            p=prob_nearest,
                        ),
                        A.Downscale(
                            scale_min=1 / scale_min,
                            scale_max=1 / scale_max,
                            interpolation=cv2.INTER_LINEAR,
                            p=1,
                        ),
                        A.Downscale(
                            scale_min=1 / scale_min,
                            scale_max=1 / scale_max,
                            interpolation=cv2.INTER_AREA,
                            p=1,
                        ),
                        A.Downscale(
                            scale_min=1 / scale_min,
                            scale_max=1 / scale_max,
                            interpolation=cv2.INTER_CUBIC,
                            p=1,
                        ),
                    ],
                    p=1,
                ),

                # -----Noise
                A.OneOf(
                    [
                        A.GaussNoise(var_limit=(1, 25), per_channel=True, p=0.4),
                        A.GaussNoise(var_limit=(1, 30), per_channel=False, p=0.2),
                        A.ISONoise(intensity=(0.05, 0.1), p=0.4),
                    ],
                    p=1,
                ),

                # -----Compression
                A.OneOf(
                    [
                        A.Compose(
                            [
                                A.ImageCompression(
                                    quality_lower=30, quality_upper=95, p=0.8
                                ),
                                A.RingingOvershoot(blur_limit=(7, 21), p=0.5),
                            ],
                            p=1,
                        ),
                        A.Compose(
                            [
                                A.RingingOvershoot(blur_limit=(7, 21), p=0.5),
                                A.ImageCompression(
                                    quality_lower=30, quality_upper=95, p=0.8
                                ),
                            ],
                            p=1,
                        ),
                    ],
                    p=1,
                ),
            ],
            p=1,
        )
        return lowres(image=self.image)["image"]

    def enhance(self, sharpen:float=0.5, blur_kernel:int=3, prob_blur:float=0.5):
        """Sharpening image to obtain the high resolution one

            Params:
            -- sharpen (int): value to choose the visibility of the sharpened image (Default: 0.5) 
            -- blur_kernel (int): kernel size for blurring the input image
            -- prob_blur (float): probability of applying the transform (Default: 0.5)

            Returns:
            -- highres_plus: high resolution image

        """
        highres_plus = A.Compose(
            [
                A.Sharpen(alpha=(sharpen, sharpen), lightness=(1,1), p=1),
                A.MedianBlur(blur_limit=(blur_kernel, blur_kernel), p=prob_blur),
            ],
            p=1,
        )
        return highres_plus(image=self.image)["image"]
