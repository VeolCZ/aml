import logging
import numpy as np
import torch
import os
from random_forests.CompositeRandomForest import CompositeRandomForest
from evaluator.Evaluator import Evaluator
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from preprocessing.TreeImageDataset import TreeImageDataset, TreePrerocessPipeline
from tboard.plotting import plot_confusion_matrix
from tboard.summarywriter import write_summary

SEED = int(os.getenv("SEED", "123"))
TEST_SIZE = float(os.getenv("TEST_SIZE", 0.1))


def train_composite() -> None:
    """
    Trains a composite Randomforest based on a predetermined dataset.
    """
    model = CompositeRandomForest()
    train_dataset = TreeImageDataset(type="eval")

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
    model.save_model("/weights/forest")


def eval_composite() -> None:
    """
    Evaluates the composite forest on several metrics including:
        Accuracy
        top 5 accuracy
        F1 score
        Multiroc
        Confusion matrix
    """
    model = CompositeRandomForest()
    model.load("/weights/forest")

    eval_dataset = TreeImageDataset(type="eval")

    all_labels = eval_dataset.get_cls_labels()
    _, test_indices, _, _ = train_test_split(
        np.arange(len(eval_dataset)),
        all_labels,
        test_size=TEST_SIZE,
        stratify=all_labels,
        random_state=SEED
    )

    test_dataset = Subset(eval_dataset, test_indices)
    x_test: list[torch.Tensor] = []
    y_test: list[torch.Tensor] = []
    z_test: list[torch.Tensor] = []

    for img, label in iter(test_dataset):
        x_test.append(img)
        one_hot_cls = torch.zeros((TreePrerocessPipeline.img_size), dtype=torch.float)
        one_hot_cls[label["cls"]] = 1.0
        y_test.append(one_hot_cls)
        z_test.append(label["bbox"])

    x = torch.stack(x_test, dim=0)
    y = torch.stack(y_test, dim=0)
    z = torch.stack(z_test, dim=0).squeeze(1)

    eval_res = Evaluator.eval(model, x, y, z)
    confusion_matrix = eval_res.confusion_matrix
    image = plot_confusion_matrix(confusion_matrix.cpu().numpy())

    writer = write_summary(run_name="Forest")
    writer.add_scalar("Classifier/Accuracy", eval_res.accuracy, 0)
    writer.add_scalar("Classifier/F1", eval_res.f1_score, 0)
    writer.add_scalar("Classifier/top_k", eval_res.top_3, 0)
    writer.add_scalar("Classifier/top_k", eval_res.top_5, 0)
    writer.add_scalar("Classifier/multiroc", eval_res.multiroc, 0)
    writer.add_image("Classifier/Confusion Matrix", image, 0)
    writer.add_scalar("Regressor/IOU", eval_res.iou, 0)
    writer.close()

    logger = logging.getLogger("Forest Eval")
    logger.info(eval_res)
