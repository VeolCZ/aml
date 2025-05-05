from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from numpy.typing import NDArray
from preprocessing.TreeImageDataset import Label
from preprocessing.TreeImageDataset import TreeImageDataset
from torch.utils.data import DataLoader


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


if __name__ == "__main__":

    train_dataloader = DataLoader(TreeImageDataset("train"))
    all_x = []
    all_y = []

    for x_batch, y_batch in train_dataloader:
        all_x.append(x_batch)
        all_y.append(y_batch)

    train_random_forest(all_x, all_y)
