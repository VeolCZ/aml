import logging
import torch
import os
from joblib import parallel_backend
from sklearn.ensemble import RandomForestRegressor
from evaluator.Evaluator import Evaluator
from random_forests.RandomForest import RandomForest
from torch.utils.data import Dataset
from preprocessing.data_util import load_data_to_mem
from tboard.summarywriter import write_summary

SEED = int(os.getenv("SEED", "123"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))


class RandomForestRegressorModel(RandomForest):

    def __init__(self) -> None:
        super().__init__(RandomForestRegressor(n_jobs=-1,
                                               random_state=SEED, n_estimators=200, min_samples_split=2,
                                               min_samples_leaf=4, max_depth=40, verbose=0, max_features="sqrt"))
        self.logger = logging.getLogger(self.__class__.__name__)

    def fit(self, train_dataset: Dataset, val_dataset: Dataset) -> None:
        """
        Fits the regressor on the training data and evaluates on validation data.

        Args:
            train_dataset (Dataset): The dataset used for training the model.
            val_dataset (Dataset): The dataset used for evaluating the model.
        """
        writer = write_summary(run_name="RandomForestRegressor_Trainer")
        x, _, z = load_data_to_mem(train_dataset)
        self.logger.info("Training started")

        with parallel_backend("loky", n_jobs=-1):
            self.model.fit(x, z)

        pred_bbox, _ = self.predict(x)
        train_iou = Evaluator.get_IOU(pred_bbox, z)
        self.logger.info(f"Training IOU: {train_iou:.4f}")
        writer.add_scalar("train/iou", train_iou, 0)

        x_val, _, z_val = load_data_to_mem(val_dataset)
        val_pred_bbox, _ = self.predict(x_val)
        val_iou = Evaluator.get_IOU(val_pred_bbox, z_val)
        self.logger.info(f"Validation IOU: {val_iou:.4f}")
        writer.add_scalar("val/iou", val_iou, 0)

    def predict(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts bounding box coordinates for the given input data.

        Args:
            data (torch.Tensor): The input data for which to make predictions.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple where the first element
                contains the predicted bounding boxes and the second is a
                placeholder tensor.
        """
        bbox = self.model.predict(data)
        return torch.tensor(bbox), torch.empty((1))
