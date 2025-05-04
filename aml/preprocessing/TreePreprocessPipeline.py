from typing import Union
from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np
import random
import torch
import cv2

SEED = 123


class TreePreprocessPipeline:
    @staticmethod
    def get_transforms() -> tuple[A.Compose, A.Compose]:
        torch.manual_seed(SEED)
        random.seed(SEED)
        np.random.seed(SEED)

        train_transforms: list[Union[A.BasicTransform, A.Affine]] = [
            A.RandomResizedCrop(scale=(0.8, 1.0), p=1.0, size=(224, 224)),
            A.HorizontalFlip(p=0.5),
            A.SafeRotate(limit=10),
            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                rotate=(-10, 10),
                p=0.75,
                border_mode=cv2.BORDER_CONSTANT,
            ),
            A.ColorJitter(p=0.8),
            A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.3), p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]

        train_preprocess = A.Compose(transforms=train_transforms,
                                     bbox_params=A.BboxParams(
                                         format="coco",
                                         label_fields=["class_labels"],
                                         min_visibility=0.2,
                                         min_area=10,
                                     )
                                     )

        eval_transforms: list[Union[A.BasicTransform, A.Affine]] = [
            A.Resize(height=256, width=256),
            A.CenterCrop(height=224, width=224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]

        eval_preprocess = A.Compose(transforms=eval_transforms,
                                    bbox_params=A.BboxParams(
                                        format="coco",
                                        label_fields=["class_labels"],
                                        min_visibility=0.2,
                                        min_area=10,
                                    )
                                    )

        return train_preprocess, eval_preprocess
