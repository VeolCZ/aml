import numpy as np
import torch
from evaluator.Evaluator import Evaluator
from random_forests.RandomForestClassifierTest import RandomForestClassifierModel
from random_forests.RandomForestRegressorTest import RandomForestRegressorModel
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from preprocessing.TreeImageDataset import TreeImageDataset


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
        test_size=0.5,
        stratify=all_labels,
        random_state=123  # ADD SEED
    )

    train_dataset_subset = Subset(train_dataset, train_indices)
    test_dataset = Subset(eval_dataset, test_indices)

    model.fit(train_dataset_subset)
    # forest.save_model(PATH)

    x_test: list[torch.Tensor] = []
    y_test: list[torch.Tensor] = []
    for img, label in iter(test_dataset):
        x_test.append(img)
        y_test.append(label["cls"])

    x = torch.stack(x_test, dim=0)
    y = torch.stack(y_test, dim=0)

    acc = Evaluator.get_accuracy(model, x, y)
    print(f"accuracy: {acc}")
    top_k = Evaluator.get_top_k(model, x, y, k=5)
    print(f"top_k:{top_k}")

    # _, cls = model.predict()
    # print(cls)


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
        test_size=0.5,
        stratify=all_labels,
        random_state=123  # ADD SEED
    )

    train_dataset_subset = Subset(train_dataset, train_indices)
    test_dataset = Subset(eval_dataset, test_indices)

    model.fit(train_dataset_subset)
    # forest.save_model(PATH)

    x_test: list[torch.Tensor] = []
    y_test: list[torch.Tensor] = []
    for img, label in iter(test_dataset):
        x_test.append(img)
        y_test.append(label["bbox"].squeeze(0))

    x = torch.stack(x_test, dim=0)
    y = torch.stack(y_test, dim=0).squeeze(0)

    # print(x)
    # print(y)

    iou = Evaluator.get_IOU(model, x, y)
    print(f"iou: {iou}")

    # bbox, _ = forest.predict([test_dataset[0][0]])
    # print(bbox)
