from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from numpy.typing import NDArray
from aml.preprocessing.TreeImageDataset import Label
from aml.preprocessing.TreeImageDataset import TreeImageDataset
from torch.utils.data import DataLoader
import torch


def train_random_forest(
    x: NDArray[np.float64], y: list[Label], test_size: float = 0.2
) -> tuple[RandomForestClassifier, float]:
    forest_classifier = RandomForestClassifier()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, stratify=y
    )
    forest_classifier.fit(x_train, y_train)
    y_pred = forest_classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: ", accuracy)
    return forest_classifier, accuracy

def my_collate_fn(data):
    images, labels = zip(*data)
    label_ids = [label.label for label in labels]
    return images, label_ids

if __name__ == "__main__":
    train_dataloader = DataLoader(TreeImageDataset("train"))
    all_x = []
    all_y = []
    print("Loading data...")
    print(type(train_dataloader))
    idx = 0
    
    to_pass = [44, 45, 46]

    for x_batch, y_batch in train_dataloader:
        print(f"Batch {idx}")
        idx += 1
        all_x.append(x_batch)
        all_y.append(y_batch)

    train_random_forest(all_x, all_y)
