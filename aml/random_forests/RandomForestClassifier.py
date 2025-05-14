from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import torch
from typing import List
from .RandomForest import RandomForest


class RandomForestClassifierModel(RandomForest):
    """
    Random Forest model for classification tasks.
    Inherits from the abstract base RandomForest class.
    """

    def _init_model(self) -> RandomForestClassifier:
        """
        Initializes the Random Forest Classifier model.
        Returns:
            RandomForestClassifier: An instance of the Random Forest Classifier.
        """
        return RandomForestClassifier()

    def _compute_metrics(self, y_test: List[torch.Tensor], y_pred: List[torch.Tensor]) -> float:
        """
        Computes the accuracy metric for classification tasks.
        Args:
            y_test (List[torch.Tensor]): List of ground truth labels.
            y_pred (List[torch.Tensor]): List of predicted labels.
        Returns:
            float: The accuracy score.
        """
        return float(accuracy_score(y_test, y_pred))

    def _get_target_key(self) -> str:
        return "cls"
