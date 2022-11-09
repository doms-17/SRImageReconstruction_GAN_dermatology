import albumentations as A
import cv2



class Augmentation:
    """Augmentation"""

    def __init__(self, image):
        self.image = image
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]

    def augment(self, p_spatial_transf=0, p_crop_resize=0, dim_crop=0):
        transform = A.Compose([
            A.Affine(scale=(1.05), keep_ratio=True,
                     shear=[-5,5], interpolation=cv2.INTER_CUBIC, mode=cv2.BORDER_CONSTANT, p=0.5),
            A.Flip(p=1),
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.2, p=1),
                A.RandomGamma(gamma_limit=(60, 140), p=1),
            ], p=1),
            A.OneOf([
                A.ElasticTransform(alpha=500, sigma=50, alpha_affine=10,
                                   interpolation=cv2.INTER_CUBIC, border_mode=cv2.BORDER_CONSTANT, p=1),
                A.GridDistortion(num_steps=20, distort_limit=0.05,
                                 interpolation=cv2.INTER_CUBIC, border_mode=cv2.BORDER_CONSTANT, p=1),
                A.NoOp(p=1),
            ], p=p_spatial_transf),
            A.Compose([
                A.CenterCrop(height=self.height-dim_crop, width=self.width-dim_crop, p=1),
                A.Resize(height=self.height, width=self.width,
                         interpolation=cv2.INTER_CUBIC, p=1),
            ], p=p_crop_resize),
        ], p=1)
        return transform(image=self.image)['image']
