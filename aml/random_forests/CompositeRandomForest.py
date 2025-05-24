import os
import torch
from random_forests.RandomForestClassifier import RandomForestClassifierModel
from random_forests.RandomForestRegressor import RandomForestRegressorModel
from interface.ModelInterface import ModelInterface
from torch.utils.data import Dataset


class CompositeRandomForest(ModelInterface):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = RandomForestClassifierModel()
        self.regressor = RandomForestRegressorModel()

    def fit(self, train_dataset: Dataset) -> None:
        self.classifier.fit(train_dataset)
        self.regressor.fit(train_dataset)

    def predict(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, cls = self.classifier.predict(data)
        bbox, _ = self.regressor.predict(data)
        return bbox, cls

    def save_model(self, path: str) -> None:
        if not os.path.isdir(path):
            os.mkdir(path)
        self.classifier.save_model(path + "/classifier.pkl")
        self.regressor.save_model(path + "/regressor.pkl")

    def load(self, path: str) -> None:
        self.classifier.load(path + "/classifier.pkl")
        self.regressor.load(path + "/regressor.pkl")
