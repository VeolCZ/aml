import os
from typing import Union
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
from preprocessing.ViTImageDataset import ViTImageDataset
from preprocessing.TreeImageDataset import TreeImageDataset


SEED = int(os.getenv("SEED", "123"))
TEST_SIZE = float(os.getenv("TEST_SIZE", 0.1))

DataType = Union[ViTImageDataset, TreeImageDataset]


def get_data_splits(train_data: DataType, test_data: DataType, seed: int = SEED, test_size: float = TEST_SIZE,
                    val_split: bool = False) -> tuple[Dataset, Dataset, Dataset]:
    all_labels = np.array((train_data.get_cls_labels()))
    raw_train_indices, test_indicies, _, _ = train_test_split(
        np.arange(len(train_data)),
        all_labels,
        test_size=test_size,
        stratify=all_labels,
        random_state=seed,
        shuffle=True
    )
    if val_split:
        train_labels = all_labels[raw_train_indices]
        train_indices, val_indicies, _, _ = train_test_split(
            np.arange(len(raw_train_indices)),
            train_labels,
            test_size=test_size,
            stratify=train_labels,
            random_state=seed,
            shuffle=True
        )

        return Subset(train_data, train_indices), Subset(test_data, val_indicies), Subset(test_data, test_indicies)
    else:
        return Subset(train_data, raw_train_indices), Subset(test_data, []), Subset(test_data, test_indicies)
