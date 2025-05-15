from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, top_k_accuracy_score
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
        #best = top_k_accuracy_score(y_test, y_pred, k=1)
        #print(f"Best accuracy: {best}")
        #top_5 = top_k_accuracy_score(y_test, y_pred, k=1)
        #print(f"Top 5 accuracy: {top_5}")
        #top_10 = top_k_accuracy_score(y_test, y_pred, k=1)
        #print(f"Top 10 accuracy: {top_10}")
        acc = float(accuracy_score(y_test, y_pred))
        return acc

    def _get_target_key(self) -> str:
        return "cls"
