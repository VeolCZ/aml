import numpy as np
import torch
from random_forests.CompositeRandomForest import CompositeRandomForest
from evaluator.Evaluator import Evaluator
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from preprocessing.TreeImageDataset import TreeImageDataset
from tboard.plotting import plot_confusion_matrix
from tboard.summarywriter import write_summary


def train_composite() -> None:
    model = CompositeRandomForest()

    train_dataset = TreeImageDataset(type="train")

    all_labels = train_dataset.get_cls_labels()
    train_indices, _, _, _ = train_test_split(
        np.arange(len(train_dataset)),
        all_labels,
        test_size=0.1,
        stratify=all_labels,
        random_state=123  # ADD SEED
    )

    train_dataset_subset = Subset(train_dataset, train_indices)

    model.fit(train_dataset_subset)
    model.save_model("/weights/forest")


def eval_composite() -> None:
    model = CompositeRandomForest()
    model.load("/weights/forest")

    train_dataset = TreeImageDataset(type="train")
    eval_dataset = TreeImageDataset(type="eval")

    all_labels = train_dataset.get_cls_labels()
    _, test_indices, _, _ = train_test_split(
        np.arange(len(train_dataset)),
        all_labels,
        test_size=0.1,
        stratify=all_labels,
        random_state=123  # ADD SEED
    )

    test_dataset = Subset(eval_dataset, test_indices)
    x_test: list[torch.Tensor] = []
    y_test: list[torch.Tensor] = []
    z_test: list[torch.Tensor] = []

    for img, label in iter(test_dataset):
        x_test.append(img)
        one_hot_cls = torch.zeros((200), dtype=torch.float)
        one_hot_cls[label["cls"]] = 1.0
        y_test.append(one_hot_cls)
        z_test.append(label["bbox"])

    x = torch.stack(x_test, dim=0)
    y = torch.stack(y_test, dim=0)
    z = torch.stack(z_test, dim=0).squeeze(1)

    print(x.shape)
    print(y.shape)
    print(z.shape)

    eval_res = Evaluator.eval(model, x, y, z)
    confusion_matrix = eval_res.confusion_matrix
    num_classes = eval_res.num_classes
    image = plot_confusion_matrix(confusion_matrix.cpu().numpy(), num_classes)

    print(eval_res)
