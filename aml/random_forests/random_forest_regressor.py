from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from numpy.typing import NDArray
from preprocessing.TreeImageDataset import TreeImageDataset
from torch.utils.data import DataLoader
import torch


def compute_many_iou(many_boxes1: list[torch.tensor], many_boxes2: list[torch.tensor]):
    """
    Compute the average Intersection over Union (IoU) of two lists of bounding boxes.
    Args:
        many_boxes1: The first list of bounding boxes (test boxes).
        many_boxes2: The second list of bounding boxes (predicted boxes).
    Returns:
        The average IoU of the two lists of bounding boxes.
    """
    i = 0
    sum = 0
    for box1, box2 in zip(many_boxes1, many_boxes2):
        sum += compute_iou(box1, box2)
        i += 1
    return sum/i


def compute_iou(box1: torch.tensor, box2: torch.tensor):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.
    Args:
        box1: The first bounding box (test).
        box2: The second bounding box (prediction).
    Returns:
        The IoU of the two bounding boxes.
    """
    # box: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    if union_area > 0:
        iou_tensor = inter_area / union_area
        return iou_tensor.item()
    else:
        return 0


def train_random_forest(
    x: NDArray[np.float64], y, test_size: float = 0.2
) -> tuple[RandomForestRegressor, float]:
    """
    Train a random forest regressor on the given data.
    Args:
        x: The input data.
        y: The labels.
        test_size: The proportion of the dataset to include in the test split.
    Returns:
        forest_regressor: The trained random forest regressor.
        iou: The IOU of the regressor on the test set.
    """
    x = [sample.flatten() for sample in x]
    x = np.array(x, dtype=np.float64)

    forest_regressor = RandomForestRegressor()
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        [label["bbox"] for label in y],
        test_size=test_size,
        stratify=[label["cls"] for label in y]
    )

    print("Fitting, this will take a while...")
    y_train = [y.squeeze() for y in y_train]
    y_test = [y.squeeze() for y in y_test]
    forest_regressor.fit(x_train, y_train)
    print(y_train)
    print("Predicting, this will take a while...")
    y_pred = forest_regressor.predict(x_test)

    print("computing IOU")
    iou = compute_many_iou(y_test, y_pred)
    print("IOU: ", iou)

    return forest_regressor, iou


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


def treereg_test() -> None:
    """
    Test the random forest regressor on the training data.
    """
    train_dataloader = DataLoader(TreeImageDataset("train"), collate_fn=my_collate_fn)
    all_x = []
    all_y = []

    for x_batch, y_batch in train_dataloader:
        all_x.extend(x_batch)
        all_y.extend(y_batch)

    train_random_forest(all_x, all_y)
