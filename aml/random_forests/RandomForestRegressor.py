from sklearn.ensemble import RandomForestRegressor
import torch
from typing import List
from .RandomForest import RandomForest


class RandomForestRegressorModel(RandomForest):
    """
    Random Forest model for regression tasks.
    Inherits from the abstract base RandomForest class.
    """

    def _init_model(self) -> RandomForestRegressor:
        """
        Initializes the Random Forest Regressor model.
        Returns:
            RandomForestRegressor: An instance of the Random Forest Regressor.
        """
        return RandomForestRegressor()

    def _compute_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> float:
        """
        Computes the Intersection over Union (IoU) of two bounding boxes.
        Args:
            box1 (torch.Tensor): The first bounding box.
            box2 (torch.Tensor): The second bounding box.
        Returns:
            float: The IoU score.
        """
        x1 = torch.max(torch.tensor([box1[0], box2[0]]))
        y1 = torch.max(torch.tensor([box1[1], box2[1]]))
        x2 = torch.min(torch.tensor([box1[2], box2[2]]))
        y2 = torch.min(torch.tensor([box1[3], box2[3]]))

        inter_area = torch.max(torch.tensor(0), x2 - x1) * torch.max(torch.tensor(0), y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        return float((inter_area / union_area).item()) if union_area > 0 else 0

    def _compute_many_iou(self, boxes1: list[torch.Tensor], boxes2: list[torch.Tensor]) -> float:
        """
        Computes the average IoU of two lists of bounding boxes.
        Args:
            boxes1 (list[torch.Tensor]): The first list of bounding boxes.
            boxes2 (list[torch.Tensor]): The second list of bounding boxes.
        Returns:
            float: The average IoU score.
        """
        assert len(boxes1) == len(boxes2)
        return sum(self._compute_iou(b1, b2) for b1, b2 in zip(boxes1, boxes2)) / len(boxes1)

    def _compute_metrics(self, y_test: List[torch.Tensor], y_pred: List[torch.Tensor]) -> float:
        """
        Computes the Intersection over Union (IoU) metric for regression tasks.
        Args:
            y_test (List[torch.Tensor]): List of ground truth bounding boxes.
            y_pred (List[torch.Tensor]): List of predicted bounding boxes.
        Returns:
            float: The average IoU score.
        """
        return self._compute_many_iou(y_test, y_pred)

    def _get_target_key(self) -> str:
        """
        Returns the target key for the regression task.
        Returns:
            str: The target key for the regression task.
        """
        return "bbox"
