import multiprocessing
import os
import numpy as np
import torch
from typing import Union
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset, DataLoader
from preprocessing.ViTImageDataset import ViTImageDataset
from preprocessing.TreeImageDataset import TreeImageDataset


SEED = int(os.getenv("SEED", "123"))
TEST_SIZE = float(os.getenv("TEST_SIZE", 0.1))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DataType = Union[ViTImageDataset, TreeImageDataset]


def get_data_splits(train_data: DataType, test_data: DataType, seed: int = SEED, test_size: float = TEST_SIZE,
                    val_split: bool = True) -> tuple[Dataset, Dataset, Dataset]:
    """
    Splits datasets into training, validation, and test sets.

    Performs a stratified split on the dataset indices to ensure class
    distribution is maintained across the splits.

    Args:
        train_data (DataType): The dataset to source the training and validation splits from.
        test_data (DataType): The dataset to source the test split from.
        seed (int): The random seed for shuffling and splitting.
        test_size (float): The proportion of data for the test/validation splits.
        val_split (bool): If True, a validation set is created from the training data.

    Returns:
        tuple[Dataset, Dataset, Dataset]: A tuple containing Subset objects for
            the train, validation, and test sets.
    """
    all_labels = np.array((train_data.get_cls_labels()), dtype=np.int16)
    raw_train_indices, test_indicies = train_test_split(
        np.arange(len(train_data)),
        test_size=test_size,
        stratify=all_labels,
        random_state=seed,
        shuffle=True
    )
    if val_split:
        train_labels = all_labels[raw_train_indices]
        train_indices, val_indicies = train_test_split(
            np.arange(len(raw_train_indices)),
            test_size=test_size,
            stratify=train_labels,
            random_state=seed,
            shuffle=True
        )

        return Subset(train_data, train_indices), Subset(test_data, val_indicies), Subset(test_data, test_indicies)
    else:
        return Subset(train_data, raw_train_indices), Subset(test_data, []), Subset(test_data, test_indicies)


def load_data_to_mem(dataset: DataType) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Loads an entire dataset into memory as tensors.

    Uses a DataLoader to iterate through all data points in a dataset and
    concatenates them into single tensors for images, labels, and bounding boxes.

    Args:
        dataset (DataType): The PyTorch Dataset to load into memory.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing
            tensors for images (x), class labels (y), and bounding boxes (z).
    """
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=multiprocessing.cpu_count(),
        pin_memory=DEVICE.type == "cuda"
    )
    x_all_batches: list[torch.Tensor] = []
    y_all_batches: list[torch.Tensor] = []
    z_all_batches: list[torch.Tensor] = []

    for _, (imgs_batch, labels_batch) in enumerate(dataloader):
        x_all_batches.append(imgs_batch)
        y_all_batches.append(labels_batch["cls"])
        z_all_batches.append(labels_batch["bbox"])

    x = torch.cat(x_all_batches, dim=0)
    y = torch.cat(y_all_batches, dim=0)
    z = torch.cat(z_all_batches, dim=0).squeeze(1)

    return x, y, z
