from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import torch
import joblib
from interface.ModelInterface import ModelInterface
from torch.utils.data import Dataset


class RandomForest(ModelInterface):
    def __init__(self, model: RandomForestClassifier | RandomForestRegressor) -> None:
        super().__init__()
        self.model = model

    def fit(self, _: Dataset) -> None:
        raise NotImplementedError("Not implemented.")

    def predict(self, _: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Not implemented.")

    def save_model(self, path: str) -> None:
        """
        Saves the trained model to the specified path.
        Args:
            path (str): The path to save the model.
        """
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        """
        Loads the model from the specified path.
        Args:
            path (str): The path to load the model from.
        """
        self.model = joblib.load(path)
