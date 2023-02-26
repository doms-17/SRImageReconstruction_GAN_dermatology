import albumentations as A
import cv2


class Augmentation:
    """Augmentation transforms"""

    def __init__(self, image):
        self.image = image
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]

    def __repr__(self) -> str:
        return "<class 'Augmentation'>"

    def __str__(self) -> str:
        return f"Image: {self.image}, Height: {self.height}, Width: {self.width}"

    def augment(self, prob_flip:float,
                prob_affine:float, 
                prob_brightContrast:float, 
                prob_spatial_transf:float,
                crop_dimension:int,
                prob_rotate:float=0,
                prob_centralCropResize:float=0):
        """Augment image using several transforms

            Params:
            -- prob_affine (float): probability of applying the transform
            -- prob_brightContrast (float): probability of applying the transform
            -- prob_spatial_transf (float): probability of applying the transform
            -- crop_dimension (int): dimension of the quantity to subtract to height and width of the image (e.g., if crop_dimension=50: image will be ceter cropped subtracting 50 pixels from each side)
            -- prob_rotate (float): probability of applying the transform (Default: 0)
            -- prob_centralCropResize (float): probability of applying the transform (Default: 0)

            Returns:
            -- augmented image
            
        """
        transform = A.Compose(
            [
            # Flip Transform
                A.OneOf(
                    [
                        A.Rotate(p=prob_rotate),
                        A.Compose([A.HorizontalFlip(p=1),A.VerticalFlip(p=1)],p=1),
                        A.HorizontalFlip(p=1),
                        A.VerticalFlip(p=1),
                        A.NoOp(p=1),
                    ], p=prob_flip,
                ),

            # Affine Transform
                A.OneOf(
                    [
                        A.Affine(
                            scale=(1.05),
                            keep_ratio=True,
                            shear=[-5, 5],
                            interpolation=cv2.INTER_CUBIC,
                            mode=cv2.BORDER_CONSTANT,
                            p=0.4,
                            ),
                        A.NoOp(p=1),
                    ], p=prob_affine,
                ),

            # Brightness-Contrast Transform
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(
                            brightness_limit=0.1,
                            contrast_limit=0.2,
                            p=1
                        ),
                        A.RandomGamma(gamma_limit=(60, 140), p=1),
                        A.NoOp(p=1),
                    ], p=prob_brightContrast,
                ),

            # Spatial Transform
                A.OneOf(
                    [
                        A.ElasticTransform(
                            alpha=500,
                            sigma=50,
                            alpha_affine=10,
                            interpolation=cv2.INTER_CUBIC,
                            border_mode=cv2.BORDER_CONSTANT,
                            p=0.4,
                        ),
                        A.GridDistortion(
                            num_steps=20,
                            distort_limit=0.05,
                            interpolation=cv2.INTER_CUBIC,
                            border_mode=cv2.BORDER_CONSTANT,
                            p=0.4,
                        ),
                        A.NoOp(p=0.2),
                    ], p=prob_spatial_transf,
                ),

            # Central Crop and Resize to original size
                A.Compose(
                    [
                        A.CenterCrop(
                            height=self.height - crop_dimension,
                            width=self.width - crop_dimension,
                            p=1,
                        ),
                        A.Resize(
                            height=self.height,
                            width=self.width,
                            interpolation=cv2.INTER_CUBIC,
                            p=1,
                        ),
                    ], p=prob_centralCropResize,
                ),

            ], p=1,
        )
        return transform(image=self.image)["image"]
