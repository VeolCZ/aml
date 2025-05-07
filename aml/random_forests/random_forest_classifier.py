from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from numpy.typing import NDArray
from preprocessing.TreeImageDataset import TreeImageDataset
from torch.utils.data import DataLoader
import torch


def train_random_forest(
    x: NDArray[np.float64], y, test_size: float = 0.2
) -> tuple[RandomForestClassifier, float]:
    forest_classifier = RandomForestClassifier()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, stratify=y
    )
    print("Fitting has begun... dun dun dunnn...")
    forest_classifier.fit(x_train, y_train)
    print("predicting, hopefully better than my future")
    y_pred = forest_classifier.predict(x_test)
    print("figuring out if this thing is better than random guessing")
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: ", accuracy)
    return forest_classifier, accuracy

def my_collate_fn(batch):
    images, labels = zip(*batch)
    return list(images), list(labels)



def tree_test():
    train_dataloader = DataLoader(TreeImageDataset("train"), collate_fn=my_collate_fn)
    all_x = []
    all_y = []
    idx = 0

    for x_batch, y_batch in train_dataloader:
        print(f"Batch {idx}")
        idx += 1
        if idx >= 5000:
            print("Karolina's computer likes being alive")
            break

        x_batch_np = np.array([x for x in x_batch], dtype=np.float64)
        y_batch_np = [int(label["cls"].view(-1)[0].item()) for label in y_batch]
        all_x.append(x_batch_np)
        all_y.extend(y_batch_np)

    all_x = np.vstack(all_x)
    all_y = np.array(all_y, dtype=np.int64)
    train_random_forest(all_x, all_y)

