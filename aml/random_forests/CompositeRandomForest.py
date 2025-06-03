import os
import torch
from random_forests.RandomForestClassifier import RandomForestClassifierModel
from random_forests.RandomForestRegressor import RandomForestRegressorModel
from interface.ModelInterface import ModelInterface
from torch.utils.data import Dataset


class CompositeRandomForest(ModelInterface):
    """
    Composite forest, contains both a Random forest classifier
    and a random forest regressor
    Attributes
        classifier: the randomforest classifier.
        regressor: the randomforest regressor.
    """

    def __init__(self) -> None:
        super().__init__()
        self.classifier = RandomForestClassifierModel()
        self.regressor = RandomForestRegressorModel()

    def fit(self, train_dataset: Dataset) -> None:
        """
        Trains both random forests
        Args:
            train_dataset(Dataset): the dataset the forest needs to be trained on.
        """
        self.classifier.fit(train_dataset)
        self.regressor.fit(train_dataset)

    def predict(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Makes a prediction from both random forests (prbability distribution of classes)
        Args:
            data(torch:Tensor): image in the form of a tensor.
        Returns:
            prediction(tuple(torch.Tensor,torch.Tensor)): First tensor contains the boundingbox prediction.
                The second tensor contains the distribution of probability of classes.
        """
        _, cls = self.classifier.predict(data)
        bbox, _ = self.regressor.predict(data)
        return bbox, cls

    def save(self, path: str) -> None:
        """
        Saves both random forests
        Args:
            path(str): the path to the directory where the forests need to be saved.
        """
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        self.classifier.save(path + "/classifier.pkl")
        self.regressor.save(path + "/regressor.pkl")

    def load(self, path: str) -> None:
        """
        loads both random forests
        Args:
            path(str): The path to the directory where the forests are saved.
        """
        self.classifier.load(path + "/classifier.pkl")
        self.regressor.load(path + "/regressor.pkl")
