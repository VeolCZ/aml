import os
import numpy as np
from random_forests.CompositeRandomForest import CompositeRandomForest
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from preprocessing.TreeImageDataset import TreeImageDataset


SEED = int(os.getenv("SEED", 123))
TEST_SIZE = 0.1


def train_composite_forest() -> None:
    assert os.path.exists("/data/CUB_200_2011"), "Please ensure the dataset is properly extracted into /data"
    assert os.path.exists("/weights"), "Please ensure the /weights directory exists"
    assert os.path.exists("/data/labels.csv"), "Please ensure the labels are generated (--make_labels)"

    PATH = "/weights/forest"
    model = CompositeRandomForest()

    train_dataset = TreeImageDataset(type="train")

    all_labels = train_dataset.get_cls_labels()
    train_indices, _, _, _ = train_test_split(
        np.arange(len(train_dataset)),
        all_labels,
        test_size=TEST_SIZE,
        stratify=all_labels,
        random_state=SEED
    )

    train_dataset_subset = Subset(train_dataset, train_indices)

    model.fit(train_dataset_subset)
    model.save_model(PATH)
