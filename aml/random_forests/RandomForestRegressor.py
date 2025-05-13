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
