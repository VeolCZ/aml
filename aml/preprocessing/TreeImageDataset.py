import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as F
from typing import Literal, TypedDict, Union
from torch.utils.data import Dataset
from torchvision.io import read_image
from preprocessing.TreePreprocessPipeline import TreePreprocessPipeline


class Label(TypedDict):
    boxes: torch.Tensor
    labels: torch.Tensor


class TreeImageDataset(Dataset):
    def __init__(
        self, type: Union[Literal["eval"], Literal["train"]]
    ) -> None:
        self._labels = pd.read_csv("/data/labels.csv")
        self._train_transform, self._eval_transform = TreePreprocessPipeline.get_transforms()
        self._type = type

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return len(self._labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, Label]:
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

        if self._type == "eval":
            transform_to_apply = self._eval_transform
        elif self._type == "train":
            transform_to_apply = self._train_transform

        transformed = transform_to_apply(
            image=image_np,
            bboxes=[bbox],
            class_labels=[label]
        )
        transformed_image = transformed["image"]
        transformed_bboxes = transformed["bboxes"]
        transformed_labels = transformed["class_labels"]

        if not transformed_bboxes:
            labels = Label(
                boxes=torch.empty((0, 4), dtype=torch.float32),
                labels=torch.empty((0,), dtype=torch.int64)
            )
        else:
            labels = Label(
                boxes=torch.tensor(transformed_bboxes, dtype=torch.float32),
                labels=torch.tensor(transformed_labels, dtype=torch.int64)
            )

        return transformed_image, labels
