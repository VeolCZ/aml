import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as F
from numpy.typing import NDArray
from preprocessing.ViTImageDataset import DatasetType, LabelType, d_type, use_cols
from preprocessing.TreePrerocessPipeline import TreePrerocessPipeline
from torch.utils.data import Dataset
from torchvision.io import read_image


class TreeImageDataset(Dataset):
    """
    Dataset for loading and preprocessing images for trees.

    Attributes:
        _labels (pd.DataFrame): DataFrame with image paths, labels, and bboxes.
        _base_transform (Callable): Transformation pipeline from TreePrerocessPipeline.
    """

    def __init__(self, type: DatasetType) -> None:
        """
        Initializes TreeImageDataset.

        Args:
            type (DatasetType): "train" or "eval", determines transformations.

        Raises:
            RuntimeError: Invalid dataset type.
        """
        self._labels = pd.read_csv("/data/labels.csv", dtype=d_type, usecols=use_cols).iloc[:540]
        if type == "eval":
            self._base_transform = TreePrerocessPipeline.get_base_eval_transform()
        elif type == "train":
            self._base_transform = TreePrerocessPipeline.get_base_train_transform()
        else:
            raise RuntimeError("Error setting transformation: Invalid dataset type")

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self._labels)

    def _transform_data(self, image_np: NDArray, label: int, bbox: list[float]) -> tuple[torch.Tensor, LabelType]:
        """
        Applies transformations to image and bounding box data.

        Args:
            image_np (NDArray): Image data as NumPy array.
            label (int): Class label.
            bbox (list[float]): [xmin, ymin, width, height] bounding box.

        Returns:
            tuple[torch.Tensor, LabelType]: Transformed image and labels ("bbox", "cls" tensors).
        """
        transformed = self._base_transform(
            image=image_np,
            bboxes=[bbox],
            class_labels=[label]
        )
        transformed_image = transformed["image"]
        transformed_bboxes = transformed["bboxes"]
        transformed_labels = int(transformed["class_labels"][0])

        labels = {
            "bbox": torch.tensor(transformed_bboxes, dtype=torch.float),
            "cls": torch.tensor(transformed_labels) - 1
        }

        return transformed_image, labels

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, LabelType]:
        """
        Retrieves image and label data for a given index.

        Reads image, transforms it, and generates tree features.

        Args:
            idx (int): Index of the data sample.

        Returns:
            tuple[NDArray, LabelType]: Image features (NumPy array) and labels.

        Raises:
            RuntimeError: On errors reading data, image, or generating features.
        """
        try:
            row = self._labels.iloc[idx]
            img_path: str = row["image_path"]
            class_id: int = row["class_id"]
            xmin = row["x"]
            ymin = row["y"]
            width = row["width"]
            height = row["height"]
            bbox: list[float] = [xmin, ymin, width, height]

        except Exception as e:
            raise RuntimeError(f"Error reading data for index {idx} from DataFrame: {e}")

        try:
            image_tensor = read_image(img_path)
            image = F.to_pil_image(image_tensor)
            image_np = np.array(image)

        except Exception as e:
            raise RuntimeError(f"Error reading/converting image at index {idx} (path: {img_path}): {e}")

        transformed_image, labels = self._transform_data(image_np, class_id, bbox)

        try:
            image_numpy_hwc = transformed_image.permute(1, 2, 0).numpy()
            image_features = TreePrerocessPipeline.tree_image_transform(image_numpy_hwc)

        except Exception as e:
            raise RuntimeError(f"Error during hog generation at index {idx}: {e}")

        return image_features, labels

    def get_cls_labels(self) -> list[int]:
        return [int(label) for label in self._labels["class_id"]]
