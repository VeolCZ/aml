import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as F
from dataclasses import dataclass
from numpy.typing import NDArray
from preprocessing.PreprocessPipeline import PreprocessPipeline
from torch.utils.data import Dataset
from torchvision.io import read_image
from typing import Literal, Union

DatasetType = Union[Literal["eval"], Literal["train"]]


@dataclass
class Label:
    bbox: torch.Tensor
    label: torch.Tensor


class TreeImageDataset(Dataset):
    def __init__(
        self, type: DatasetType
    ) -> None:
        self._labels = pd.read_csv("/data/labels.csv")
        if type == "eval":
            self._base_transform = PreprocessPipeline.get_base_eval_transform()
        elif type == "train":
            self._base_transform = PreprocessPipeline.get_base_train_transform()
        else:
            raise RuntimeError("Error setting transformation: Invalid dataset type")

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return len(self._labels)

    def _transform_data(self, image_np: NDArray, label: int, bbox: list[float]) -> tuple[torch.Tensor, Label]:
        transformed = self._base_transform(
            image=image_np,
            bboxes=[bbox],
            class_labels=[label]
        )
        transformed_image = transformed["image"]
        transformed_bboxes = transformed["bboxes"]
        transformed_labels = transformed["class_labels"]

        if not transformed_bboxes:
            labels = Label(
                bbox=torch.empty((0, 4), dtype=torch.float32),
                label=torch.empty((0,), dtype=torch.int64)
            )
        else:
            labels = Label(
                bbox=torch.tensor(transformed_bboxes, dtype=torch.float32),
                label=torch.tensor(transformed_labels, dtype=torch.int16)
            )

        return transformed_image, labels

    def __getitem__(self, idx: int) -> tuple[NDArray, Label]:
        try:
            row = self._labels.iloc[idx]
            img_path_value = row["image_path"]
            class_id_value = row["class_id"]
            xmin = row["x"]
            ymin = row["y"]
            width = row["width"]
            height = row["height"]

            img_path = str(img_path_value)
            label = int(class_id_value)
            bbox = [float(xmin), float(ymin), float(width), float(height)]

        except Exception as e:
            raise RuntimeError(f"Error reading data for index {idx} from DataFrame: {e}")

        try:
            image_tensor = read_image(img_path)
            image = F.to_pil_image(image_tensor)
            image_np = np.array(image)
        except Exception as e:
            raise RuntimeError(f"Error reading/converting image at index {idx} (path: {img_path}): {e}")

        transformed_image, labels = self._transform_data(image_np, label, bbox)

        try:
            image_numpy_hwc = transformed_image.permute(1, 2, 0).numpy()
            image_features = PreprocessPipeline.tree_image_transform(image_numpy_hwc)

        except Exception as e:
            raise RuntimeError(f"Error during hog generation at index {idx}: {e}")

        return image_features, labels
