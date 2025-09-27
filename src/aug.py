import albumentations as A
from .config import IMG_SIZE

def train_tfms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.3),
        A.Affine(translate_percent=(0.0, 0.02), scale=(0.9, 1.1), rotate=(-10, 10), p=0.5),
        # Keep parameters minimal to satisfy different Albumentations versions
        A.CoarseDropout(max_holes=1, max_height=IMG_SIZE//10, max_width=IMG_SIZE//10, p=0.2),
        A.Resize(IMG_SIZE, IMG_SIZE),
    ])

def val_tfms():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
    ])
