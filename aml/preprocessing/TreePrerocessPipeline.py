import os
import albumentations as A
import numpy as np
import random
import torch
import cv2
from numpy.typing import NDArray
from skimage.feature import hog
from albumentations.pytorch import ToTensorV2
from typing import Union

SEED = int(os.getenv("SEED", 123))


class TreePrerocessPipeline:
    """
    Provides static methods for defining image transformation pipelines for tree image data.
    """
    img_size = 224

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
            A.RandomResizedCrop(scale=(0.8, 1.0), p=1.0, size=(
                TreePrerocessPipeline.img_size, TreePrerocessPipeline.img_size)),
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

            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
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
            A.CenterCrop(height=TreePrerocessPipeline.img_size, width=TreePrerocessPipeline.img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]

        return A.Compose(transforms=eval_transforms,
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
    def tree_image_transform(image: NDArray) -> torch.Tensor:
        """
        Extracts HOG features from an image.

        Args:
            image (NDArray): The input image as a NumPy array (HWC, RGB).

        Returns:
            NDArray[np.float64]: The HOG features as a NumPy array.
        """
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        features = hog(img_gray,
                       orientations=9,
                       pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2),
                       block_norm="L2-Hys",
                       feature_vector=True)

        return torch.tensor(features)

    @staticmethod
    def tree_predict_transform(image: NDArray) -> torch.Tensor:
        transform = TreePrerocessPipeline.get_base_eval_transform()
        raw_img = transform(
            image=image,
            bboxes=[],
            class_labels=[])["image"]
        trans_image = raw_img.permute(1, 2, 0).numpy()
        features = TreePrerocessPipeline.tree_image_transform(trans_image)
        return features.reshape(1, -1)
