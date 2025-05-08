from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from numpy.typing import NDArray
from preprocessing.TreeImageDataset import TreeImageDataset
from torch.utils.data import DataLoader
import torch


def train_random_forest(
    x: NDArray[np.float64], y: np.ndarray[int], test_size: float = 0.2
) -> tuple[RandomForestClassifier, float]:
    """
    Train a random forest classifier on the given data.
    Args:
        x: The input data.
        y: The labels.
        test_size: The proportion of the dataset to include in the test split.
    Returns:
        forest_classifier: The trained random forest classifier.
        accuracy: The accuracy of the classifier on the test set.
    """
    forest_classifier = RandomForestClassifier()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, stratify=y
    )
    print("Fitting, this will take a while...")
    forest_classifier.fit(x_train, y_train)
    print("Predicting, this will take a while...")
    y_pred = forest_classifier.predict(x_test)
    print("Computing accuracy")
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: ", accuracy)
    return forest_classifier, accuracy


def my_collate_fn(batch: list[tuple[torch.Tensor, dict]]) -> tuple[list[torch.Tensor], list[dict]]:
    """
    Custom collate function to handle the batch of data.
    Args:
        batch: A list of tuples containing the image tensor and the corresponding label dictionary.
    Returns:
        A tuple containing a list of image tensors and a list of label dictionaries.
    """
    images, labels = zip(*batch)
    return list(images), list(labels)


def treecls_test() -> None:
    """
    Test the random forest classifier on the training data.
    """
    train_dataloader = DataLoader(TreeImageDataset("train"), collate_fn=my_collate_fn)
    all_x = []
    all_y = []

    for x_batch, y_batch in train_dataloader:
        all_x.append(x_batch)
        all_y.extend([int(label["cls"]) for label in y_batch])

    all_x = np.vstack(all_x)
    all_y = np.array(all_y, dtype=np.int64)
    train_random_forest(all_x, all_y)
