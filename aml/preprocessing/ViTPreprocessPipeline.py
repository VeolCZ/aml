import os
import albumentations as A
import torch
import cv2
from albumentations.pytorch import ToTensorV2
from typing import Union, Literal
from transformers import ViTImageProcessor
from transformers.feature_extraction_sequence_utils import BatchFeature
from numpy.typing import NDArray

SEED = int(os.getenv("SEED", "123"))
robustness_type = Union[Literal["gaussian"], Literal["saltandpepper"], Literal["motionblur"], Literal["superpixels"]]


class ViTPreprocessPipeline:
    """
    Provides static methods for defining image transformation pipelines for ViT image data.
    """

    # static prop
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224", cache_dir="/data/vit_preprocess")
    processor.do_rescale = False
    processor.do_resize = False
    processor.do_normalize = False
    processor.do_convert_rgb = False
    img_size = 224

    @staticmethod
    def get_base_train_transform() -> A.Compose:
        """
        Returns the base transformation pipeline for training.

        Applies a series of augmentations, normalization, and tensor conversion.

        Returns:
            A.Compose: The training transformation pipeline.
        """
        train_transforms: list[Union[A.BasicTransform, A.Affine, A.BaseCompose]] = [
            A.RandomResizedCrop(
                scale=(0.8, 1.0),
                p=1.0,
                size=(ViTPreprocessPipeline.img_size, ViTPreprocessPipeline.img_size),
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.SafeRotate(limit=10, border_mode=cv2.BORDER_CONSTANT),
            A.SomeOf(
                [
                    A.Downscale(
                        scale_range=(0.8, 0.9),
                        interpolation_pair={"upscale": 0, "downscale": 0},
                        p=0.2,
                    ),
                    A.ColorJitter(p=0.3, hue=(-0.04, 0.04)),
                    A.MotionBlur(p=0.4),
                    A.RandomBrightnessContrast(p=0.4),
                    A.GaussNoise(p=0.1),
                ],
                p=1,
                n=2,
                replace=False,
            ),
            A.Normalize(),
            ToTensorV2(),
        ]
        return A.Compose(
            transforms=train_transforms,
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["class_labels"],
                clip=True,
            ),
            seed=SEED,
        )

    @staticmethod
    def get_base_eval_transform() -> A.Compose:
        """
        Returns the base transformation pipeline for evaluation.

        Args:
            gaussian_noise_severity (float): the severity of the gaussian noise to be applied on the image

        Returns:
            A.Compose: The evaluation transformation pipeline.
        """
        eval_transforms: list[Union[A.BasicTransform, A.Affine]] = [
            A.Resize(
                height=ViTPreprocessPipeline.img_size,
                width=ViTPreprocessPipeline.img_size,
            ),
            A.Normalize(),
            ToTensorV2(),
        ]
        return A.Compose(
            transforms=eval_transforms,
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"], clip=True),
            seed=SEED,
        )

    @staticmethod
    def get_base_robustness_transform(severity: float, alteration_type: str) -> A.Compose:
        """
        Returns the base transformation pipeline for evaluation.

        Args:
            severity (float): the severity of the noise to be applied on the image

        Returns:
            A.Compose: The evaluation transformation pipeline.
        """
        alteration_method = None
        match alteration_type:
            case "gaussian":
                alteration_method = A.GaussNoise(std_range=(severity, severity), p=1)
            case "saltandpepper":
                alteration_method = A.SaltAndPepper(amount=(severity, severity), p=1)
            case "motionblur":
                alteration_method = A.MotionBlur(blur_limit=(int(100 * severity), int(100 * severity)), p=1)
            case "superpixels":
                alteration_method = A.Superpixels(p_replace=(severity, severity), p=1)
            case _:
                raise ValueError(f"{alteration_type} is not an accepted alteration type")

        eval_transforms: list[Union[A.BasicTransform, A.Affine]] = [
            A.Resize(
                height=ViTPreprocessPipeline.img_size,
                width=ViTPreprocessPipeline.img_size,
            ),
            alteration_method,
            A.Normalize(),
            ToTensorV2(),
        ]

        return A.Compose(
            transforms=eval_transforms,
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["class_labels"],
                min_visibility=0.8,
                clip=True,
            ),
            seed=SEED,
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

    @staticmethod
    def vit_predict_transform(image: NDArray) -> torch.Tensor:
        """
        Prepares a raw image for prediction with a ViT model.

        This applies the standard evaluation pipeline and ViT-specific processing
        to a single image, returning a batched tensor ready for inference.

        Args:
            image (NDArray): The input image as a NumPy array.

        Returns:
            torch.Tensor: The processed image tensor.
        """
        transform = ViTPreprocessPipeline.get_base_eval_transform()
        raw_img = transform(image=image, bboxes=[], class_labels=[])["image"]
        features = ViTPreprocessPipeline.vit_image_transform(raw_img)
        return torch.tensor(features.pixel_values[0]).unsqueeze(0)
