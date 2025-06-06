import abc
import torch
from torch.utils.data import Dataset


class ModelInterface(abc.ABC):
    @abc.abstractmethod
    def fit(self, train_dataset: Dataset, val_dataset: Dataset) -> None:
        """Trains the model on the provided training and validation datasets.

        Args:
            train_dataset (Dataset): The dataset used for training the model.
            val_dataset (Dataset): The dataset used for validation during training.
        """
        raise NotImplementedError("Not implemented.")

    @abc.abstractmethod
    def predict(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Generates predictions for a batch of input data.

        Args:
            data (torch.Tensor): The input tensor for which to make predictions.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the predicted bounding
            box and the predicted class probabilities.
        """
        raise NotImplementedError("Not implemented.")

    @abc.abstractmethod
    def load(self, path: str) -> None:
        """Loads model weights and configuration from a file.

        Args:
            path (str): The file path from which to load the model state.
        """
        raise NotImplementedError("Not implemented.")
