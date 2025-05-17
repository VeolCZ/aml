from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, top_k_accuracy_score
import torch
from typing import List
from .RandomForest import RandomForest
import numpy as np


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

    def _compute_metrics(self, y_test: List[torch.Tensor], y_pred: tuple[torch.Tensor, torch.Tensor]) -> float:
        """
        Computes the accuracy metric for classification tasks.
        Args:
            y_test (List[torch.Tensor]): List of ground truth labels.
            y_pred (List[torch.Tensor]): List of predicted labels.
        Returns:
            float: The accuracy score.
        """
        # cls_label= [x for x in range(1,201)] # this one is for running the full dataset
        # test_cls_label = [x for x in range(1,7)] # adjust the number based on the number of classes provided in image_class_labels + 1
        # best = top_k_accuracy_score(y_test, y_pred, k=1,labels= test_cls_label)
        # print(f"Best accuracy: {best}")
        # top_5 = top_k_accuracy_score(y_test, y_pred, k=5,labels= test_cls_label)
        # print(f"Top 5 accuracy: {top_5}")
        # top_10 = top_k_accuracy_score(y_test, y_pred, k=10,labels= test_cls_label)
        # print(f"Top 10 accuracy: {top_10}")
        acc = float(accuracy_score(y_test, y_pred))
        return acc

    def find_top_k(self, y_test: List[torch.Tensor], x: List[torch.Tensor]) -> float:
        y_test_array = torch.stack(y_test).numpy() if isinstance(y_test[0], torch.Tensor) else np.array(y_test)

        if y_test_array.ndim == 2:
            print("Detected one-hot encoded y_test. Converting to class indices...")
            y_test_array = np.argmax(y_test_array, axis=1)
        else:
            y_test_array = y_test_array.squeeze()

        x_array = np.stack([sample.numpy() if isinstance(sample, torch.Tensor) else sample for sample in x])
        y_pred_array = self.model.predict_proba(x_array)

        if isinstance(y_pred_array, list):
            print("Converting list to ndarray via np.array...")
            y_pred_array = np.array(y_pred_array)

        if y_pred_array.ndim == 3:
            print("Detected 3D array, reshaping...")
            y_pred_array = np.transpose(y_pred_array, (1, 0, 2))
            y_pred_array = np.squeeze(y_pred_array)

        print("y_test shape:", y_test_array.shape)
        print("y_pred shape:", y_pred_array.shape)

        test_cls_label = list(range(y_pred_array.shape[1]))  # automatically set label range

        top_1 = top_k_accuracy_score(y_test_array, y_pred_array, k=1, labels=test_cls_label)
        print(f"Top-1 accuracy: {top_1}")

        return top_1

    def _get_target_key(self) -> str:
        return "cls"
