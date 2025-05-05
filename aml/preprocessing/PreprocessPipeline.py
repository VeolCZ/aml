import albumentations as A
import numpy as np
import random
import torch
import cv2
from numpy.typing import NDArray
from skimage.feature import hog
from albumentations.pytorch import ToTensorV2
from typing import Union

SEED = 123


class PreprocessPipeline:
    @staticmethod
    def get_base_train_transform() -> A.Compose:
        torch.manual_seed(SEED)
        random.seed(SEED)
        np.random.seed(SEED)

        train_transforms: list[Union[A.BasicTransform, A.Affine]] = [
            A.Downscale(scale_range=(0.8, 0.9), interpolation_pair={"upscale": 0, "downscale": 0}, p=0.1),
            A.RandomResizedCrop(scale=(0.8, 1.0), p=1.0, size=(224, 224)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.SafeRotate(limit=10),
            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                rotate=(-10, 10),
                p=0.75,
                border_mode=cv2.BORDER_CONSTANT,
            ),
            A.ColorJitter(p=0.8, hue=(-0.04, 0.04)),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]

        return A.Compose(transforms=train_transforms,
                         bbox_params=A.BboxParams(
                             format="coco",
                             label_fields=["class_labels"],
                             min_visibility=0.2,
                             min_area=10,
                         )
                         )

    @staticmethod
    def get_base_eval_transform() -> A.Compose:
        torch.manual_seed(SEED)
        random.seed(SEED)
        np.random.seed(SEED)

        eval_transforms: list[Union[A.BasicTransform, A.Affine]] = [
            A.Resize(height=256, width=256),
            A.CenterCrop(height=224, width=224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]

        return A.Compose(transforms=eval_transforms,
                         bbox_params=A.BboxParams(
                             format="coco",
                             label_fields=["class_labels"],
                             min_visibility=0.2,
                             min_area=10,
                         )
                         )

    @staticmethod
    def tree_image_transform(image: NDArray) -> NDArray[np.float64]:
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        features = hog(img_gray,
                       orientations=9,
                       pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2),
                       block_norm="L2-Hys",
                       feature_vector=True)

        return np.asarray(features, dtype=np.float64)
