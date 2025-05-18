import abc
import torch
from torch.utils.data import Dataset


class ModelInterface(abc.ABC):
    @abc.abstractmethod
    def fit(self, train_dataset: Dataset) -> None:
        raise NotImplementedError("Not implemented.")

    @abc.abstractmethod
    def predict(self, data: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Not implemented.")
