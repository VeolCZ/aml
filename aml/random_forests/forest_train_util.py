import numpy as np
import torch
from evaluator.Evaluator import Evaluator
from random_forests.RandomForestClassifier import RandomForestClassifierModel
from random_forests.RandomForestRegressor import RandomForestRegressorModel
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from preprocessing.TreeImageDataset import TreeImageDataset
from tboard.plotting import plot_confusion_matrix
from tboard.summarywriter import write_summary


def train_forests() -> dict[str:float]:
    writer = write_summary(run_name="aml/runs/random_forest_thing")
    cls_results = train_classifier_forest(writer)
    reg_results = train_regressor_forest(writer)
    writer.close()
    return cls_results, reg_results


def train_classifier_forest(writer:bool =True) -> dict[str:float]:
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
    y = torch.stack(y_test, dim=1)

    eval = Evaluator.classifier_eval(model, x, y)
    print(f"Evaluation scores: {eval}")
    confusion_matrix = eval["confusion_matrix"]
    num_classes = eval["num_classes"]
    image = plot_confusion_matrix(confusion_matrix.cpu().numpy(), num_classes)
    if writer:
        write_summary().add_scalar("Classifier/Accuracy", eval["accuracy"], 0)
        write_summary().add_scalar("Classifier/F1", eval["f1_score"], 0)
        write_summary().add_scalar("Classifier/top_k", eval["top_k"], 0)
        write_summary().add_scalar("Classifier/multiroc", eval["multiroc"], 0)
        write_summary().add_image("Classifier/Confusion Matrix", image, 0)
    return eval


def train_regressor_forest(writer:bool =True) -> float:
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

    iou = Evaluator.get_IOU(model, x, y)
    print(f"iou: {iou}")
    if writer:
        write_summary().add_scalar("Regressor/IOU", iou, 0)
    return iou
