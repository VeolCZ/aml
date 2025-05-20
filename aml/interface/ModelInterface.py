import abc
import torch
from torch.utils.data import Dataset


class ModelInterface(abc.ABC):
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError("Not implemented.")
    @abc.abstractmethod
    def fit(self, train_dataset: Dataset) -> None:
        raise NotImplementedError("Not implemented.")

    @abc.abstractmethod
    def predict(self, data: torch.Tensor, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Not implemented.")

    @abc.abstractmethod
    def load(self, path: str) -> None:
        raise NotImplementedError("Not implemented.")
