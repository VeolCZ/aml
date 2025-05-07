import os
import albumentations as A
import numpy as np
import random
import torch
import cv2
from albumentations.pytorch import ToTensorV2
from typing import Union
from transformers import ViTImageProcessor, BatchFeature

SEED = int(os.getenv("SEED", 123))


class ViTPreprocessPipeline:
    """
    Provides static methods for defining image transformation pipelines for ViT image data.
    """

    # static prop
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224", cache_dir="/data/vit_preprocess")
    processor.do_rescale = False

    @staticmethod
    def get_base_train_transform() -> A.Compose:
        """
        Returns the base transformation pipeline for training.

        Applies a series of augmentations, normalization, and tensor conversion.

        Returns:
            A.Compose: The training transformation pipeline.
        """
        torch.manual_seed(SEED)
        random.seed(SEED)
        np.random.seed(SEED)

        train_transforms: list[Union[A.BasicTransform, A.Affine, A.BaseCompose]] = [
            A.RandomResizedCrop(scale=(0.8, 1.0), p=1.0, size=(224, 224)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.SafeRotate(limit=10, border_mode=cv2.BORDER_CONSTANT),

            A.SomeOf([
                A.Downscale(scale_range=(0.8, 0.9), interpolation_pair={"upscale": 0, "downscale": 0}, p=0.2),
                A.ColorJitter(p=0.3, hue=(-0.04, 0.04)),
                A.MotionBlur(p=0.4),
                A.RandomBrightnessContrast(p=0.4),
                A.GaussNoise(p=0.1),
                A.RandomRain(p=0.05)
            ], p=1, n=2, replace=False),

            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), normalization="min_max"),
            ToTensorV2(),
        ]

        return A.Compose(transforms=train_transforms,
                         bbox_params=A.BboxParams(
                             format="coco",
                             label_fields=["class_labels"],
                             min_visibility=0.2,
                             min_area=10,
                             clip=True
                         ),
                         seed=SEED
                         )

    @staticmethod
    def get_base_eval_transform() -> A.Compose:
        """
        Returns the base transformation pipeline for evaluation.

        Returns:
            A.Compose: The evaluation transformation pipeline.
        """
        torch.manual_seed(SEED)
        random.seed(SEED)
        np.random.seed(SEED)

        eval_transforms: list[Union[A.BasicTransform, A.Affine]] = [
            A.Resize(height=256, width=256),
            A.CenterCrop(height=224, width=224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), normalization="min_max"),
            ToTensorV2(),
        ]

        return A.Compose(transforms=eval_transforms,
                         bbox_params=A.BboxParams(
                             format="coco",
                             label_fields=["class_labels"],
                             min_visibility=0.2,
                             min_area=10,
                             clip=True
                         )
                         )

    @staticmethod
    def vit_image_transform(image: torch.Tensor) -> BatchFeature:
        """
        Transforms an image tensor using the ViT processor.

        Args:
            image (torch.Tensor): The input image as a PyTorch tensor.

        Returns:
            BatchFeature: The processed image features.
        """
        return ViTPreprocessPipeline.processor(image)
