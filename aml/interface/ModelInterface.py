import abc
import torch
from torch.utils.data import Dataset


class ModelInterface(abc.ABC):
    @abc.abstractmethod
    def fit(self, train_dataset: Dataset) -> None:
        raise NotImplementedError("Not implemented.")

    @abc.abstractmethod
    def predict(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # if the model only predicts one tensor return thst and other one empty
        raise NotImplementedError("Not implemented.")

    @abc.abstractmethod
    def load(self, path: str) -> None:
        raise NotImplementedError("Not implemented.")
