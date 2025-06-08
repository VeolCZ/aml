from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import torch
import joblib
from interface.ModelInterface import ModelInterface
from torch.utils.data import Dataset


class RandomForest(ModelInterface):
    """A wrapper for scikit-learn models to conform to the ModelInterface."""

    def __init__(self, model: RandomForestClassifier | RandomForestRegressor) -> None:
        """
        Initializes the RandomForest model wrapper.

        Args:
            model (RandomForestClassifier | RandomForestRegressor): An instance
                of a scikit-learn Random Forest model.
        """
        super().__init__()
        self.model = model

    def fit(self, _: Dataset) -> None:
        """
        This method is not implemented for the base RandomForest class.
        """
        raise NotImplementedError("Not implemented.")

    def predict(self, _: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This method is not implemented for the base RandomForest class.
        """
        raise NotImplementedError("Not implemented.")

    def save(self, path: str) -> None:
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
