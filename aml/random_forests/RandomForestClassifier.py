import logging
import torch
import os
from sklearn.ensemble import RandomForestClassifier
from random_forests.RandomForest import RandomForest
from torch.utils.data import Dataset
from joblib import parallel_backend
from preprocessing.data_util import load_data_to_mem
from evaluator.Evaluator import Evaluator
from tboard.summarywriter import write_summary

SEED = int(os.getenv("SEED", "123"))


class RandomForestClassifierModel(RandomForest):

    def __init__(self) -> None:
        super().__init__(RandomForestClassifier(n_jobs=-1,
                                                random_state=SEED, n_estimators=200, min_samples_split=2,
                                                min_samples_leaf=4, max_depth=40, verbose=0, max_features="sqrt"))
        self.logger = logging.getLogger(self.__class__.__name__)

    def fit(self, train_dataset: Dataset, val_dataset: Dataset) -> None:
        """
        Fits the classifier on the training data and evaluates on validation data.

        Args:
            train_dataset (Dataset): The dataset used for training the model.
            val_dataset (Dataset): The dataset used for evaluating the model.
        """
        writer = write_summary(run_name="RandomForestClassifier_Trainer")
        x, y_one_hot, _ = load_data_to_mem(train_dataset)
        y = y_one_hot.argmax(-1)
        self.logger.info("Training started")

        with parallel_backend("loky", n_jobs=-1):
            self.model.fit(x, y)

        _, pred_cls = self.predict(x)
        train_accuracy = Evaluator.get_accuracy(y_one_hot, pred_cls.argmax(-1))
        self.logger.info(f"Training Accuracy: {train_accuracy:.4f}")
        writer.add_scalar("train/accuracy", train_accuracy, 0)

        x_val, y_val, _ = load_data_to_mem(val_dataset)
        _, val_pred_cls = self.predict(x_val)
        val_accuracy = Evaluator.get_accuracy(y_val, val_pred_cls.argmax(-1))
        self.logger.info(f"Validation Accuracy: {val_accuracy:.4f}")
        writer.add_scalar("val/accuracy", val_accuracy, 0)

    def predict(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts class probabilities for the given input data.

        Args:
            data (torch.Tensor): The input data for which to make predictions.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple where the first element
                is a placeholder tensor and the second contains the predicted
                class probabilities.
        """
        cls = self.model.predict_proba(data)

        return torch.empty((1)), torch.tensor(cls)
