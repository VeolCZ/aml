from typing import Literal
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from numpy.typing import NDArray
from preprocessing.TreeImageDataset import TreeImageDataset
from torch.utils.data import DataLoader
import torch

TaskType = Literal["classification", "regression"]


class RandomForest:

    def __init__(self, task_type: TaskType):
        self.task_type = task_type
        if task_type == "classification":
            self.model = RandomForestClassifier()
        elif task_type == "regression":
            self.model = RandomForestRegressor()
        else:
            raise ValueError(
                f"Unsupported task_type type {task_type}. Should be 'classification' or 'regression'."
            )

    def _compute_iou(self, box1: torch.tensor, box2: torch.tensor):
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

    def _compute_many_iou(self, boxes1: list[torch.tensor], boxes2: list[torch.tensor]):
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
        for box1, box2 in zip(boxes1, boxes2):
            sum += self._compute_iou(box1, box2)
            i += 1
        return sum / i

    def _compute_metrics(self, y_test, y_pred) -> float:
        if self.task_type == "classification":
            return self._compute_many_iou(y_test, y_pred)
        elif self.task_type == "regression":
            return accuracy_score(y_test, y_pred)

    def _collate(self, batch):
        images, labels = zip(*batch)
        return list(images), list(labels)

    def _load_data(self):
        train_dataloader = DataLoader(
            TreeImageDataset("train"), collate_fn=self._collate
        )
        self.x = []
        self.y = []

        for x_batch, y_batch in train_dataloader:
            self.x.extend(x_batch)
            self.y.extend(y_batch)

    def train(self, test_size=0.2):
        if self.task_type == "regression":
            target = "cls"
        elif self.task_type == "classification":
            target = "bbox"

        x = [sample.flatten() for sample in self.x]
        x = np.array(x, dtype=np.float64)
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            [label[target] for label in self.y],
            test_size=test_size,
            stratify=[label["cls"] for label in self.y],
        )

        print("Fitting, this will take a while...")
        y_train = [y.squeeze() for y in y_train]
        y_test = [y.squeeze() for y in y_test]
        self.model.fit(x_train, y_train)
        print(y_train)
        print("Predicting, this will take a while...")
        y_pred = self.model.predict(x_test)
        eval_metric = self._compute_metrics(y_test, y_pred)
        return eval_metric

    def process(self):
        self._load_data()
        self.train()
