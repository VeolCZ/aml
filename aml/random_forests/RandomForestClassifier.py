import logging
import torch
import os
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from random_forests.RandomForest import RandomForest
from torch.utils.data import Dataset
from joblib import parallel_backend
from preprocessing.data_util import load_data_to_mem
from evaluator.Evaluator import Evaluator

SEED = int(os.getenv("SEED", "123"))


class RandomForestClassifierModel(RandomForest):
    """
    Random Forest model for classification tasks.
    Inherits from the abstract base RandomForest class.
    """

    def __init__(self) -> None:
        super().__init__(RandomForestClassifier(n_jobs=-1,
                                                random_state=SEED, n_estimators=200, min_samples_split=2,
                                                min_samples_leaf=4, max_depth=40, verbose=0, max_features="sqrt"))
        self.logger = logging.getLogger(self.__class__.__name__)

    def fit(self, train_dataset: Dataset, val_dataset: Dataset) -> None:
        """
        Trains the model
        Args:
            train_dataset(Dataset): the dataset the forest needs to be trained on.
        """
        x, y_one_hot, _ = load_data_to_mem(train_dataset)
        y = y_one_hot.argmax(-1)
        self.logger.info("Training started")

        with parallel_backend("loky", n_jobs=-1):
            self.model.fit(x, y)

        _, pred_cls = self.predict(x)
        train_accuracy = Evaluator.get_accuracy(y_one_hot, pred_cls.argmax(-1))
        self.logger.info(f"Training Accuracy: {train_accuracy:.4f}")

        x_val, y_val, _ = load_data_to_mem(val_dataset)
        _, val_pred_cls = self.predict(x_val)
        val_accuracy = Evaluator.get_accuracy(y_val, val_pred_cls.argmax(-1))
        self.logger.info(f"Training Accuracy: {val_accuracy:.4f}")

    def predict(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Makes a prediction from the model(prbability distribution of classes)
        Args:
            data(torch:Tensor): image in the form of a tensor.
        Returns:
            prediction(tuple(torch.Tensor,torch.Tensor)): first tensor is empty
                the second tensor contains the distribution of probability of classes.
        """
        cls = self.model.predict_proba(data)

        return torch.empty((1)), torch.tensor(cls)
