import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transformations(max_bright_contrast: float = 0.25, probability: float = 0.50) -> tuple[A.Compose, A.Compose]:
    """
    Creates transformations for Data Augmentation while still mainting realistic geographical data.
    """
    transform_train = A.Compose([
        A.HorizontalFlip(p = probability),
        A.VerticalFlip(p = probability),
        A.RandomRotate90(p = probability),
        A.RandomBrightnessContrast(
            brightness_limit = max_bright_contrast,
            contrast_limit = max_bright_contrast,
            p = probability
        ),
        ToTensorV2()
    ])
        
    # Validation data can't be transformed
    transform_val = A.Compose([ToTensorV2()])
        
    return transform_train, transform_val