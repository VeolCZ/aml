import os
import numpy as np
from random_forests.CompositeRandomForest import CompositeRandomForest
from random_forests.RandomForestClassifier import RandomForestClassifierModel
from random_forests.RandomForestRegressor import RandomForestRegressorModel
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from preprocessing.TreeImageDataset import TreeImageDataset


SEED = int(os.getenv("SEED", 123))


def train_classifier_forest() -> None:
    """
    Run the random forest training for classification.
    """
    PATH = "/logs/forest_classifier.pkl"
    model = RandomForestClassifierModel()

    train_dataset = TreeImageDataset(type="train")
    eval_dataset = TreeImageDataset(type="eval")

    all_labels = train_dataset.get_cls_labels()
    train_indices, test_indices, _, _ = train_test_split(
        np.arange(len(train_dataset)),
        all_labels,
        test_size=0.1,
        stratify=all_labels,
        random_state=SEED
    )

    train_dataset_subset = Subset(train_dataset, train_indices)
    _test_dataset = Subset(eval_dataset, test_indices)

    model.fit(train_dataset_subset)
    model.save_model(PATH)


def train_regressor_forest() -> None:
    """
    Run the random forest training for regression.
    """
    PATH = "/logs/forest_regressor.pkl"
    model = RandomForestRegressorModel()

    train_dataset = TreeImageDataset(type="train")
    eval_dataset = TreeImageDataset(type="eval")

    all_labels = train_dataset.get_cls_labels()
    train_indices, test_indices, _, _ = train_test_split(
        np.arange(len(train_dataset)),
        all_labels,
        test_size=0.1,
        stratify=all_labels,
        random_state=SEED
    )

    train_dataset_subset = Subset(train_dataset, train_indices)
    _test_dataset = Subset(eval_dataset, test_indices)

    model.fit(train_dataset_subset)
    model.save_model(PATH)


def train_composite_forest() -> None:
    PATH = "/weights/forest"
    model = CompositeRandomForest()

    train_dataset = TreeImageDataset(type="train")
    eval_dataset = TreeImageDataset(type="eval")

    all_labels = train_dataset.get_cls_labels()
    train_indices, test_indices, _, _ = train_test_split(
        np.arange(len(train_dataset)),
        all_labels,
        test_size=0.1,
        stratify=all_labels,
        random_state=SEED
    )

    train_dataset_subset = Subset(train_dataset, train_indices)
    _test_dataset = Subset(eval_dataset, test_indices)

    model.fit(train_dataset_subset)
    model.save_model(PATH)
