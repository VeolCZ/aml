from sklearn.ensemble import RandomForestRegressor
import torch
from random_forests.RandomForestTest import RandomForestTest
from torch.utils.data import Dataset


class RandomForestRegressorModel(RandomForestTest):
    """
    Random Forest model for classification tasks.
    Inherits from the abstract base RandomForest class.
    """

    def __init__(self) -> None:
        super().__init__(RandomForestRegressor())

    def name(self) -> str:
        return "RandomForestRegressor"

    def fit(self, train_dataset: Dataset) -> None:
        x_train, y_train = [], []
        for i in range(len(train_dataset)):
            img, label = train_dataset[i]
            x_train.append(img)
            y_train.append(label["bbox"].tolist()[0])

        self.model.fit(x_train, y_train)

    def predict(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bbox = self.model.predict(data)
        return torch.tensor(bbox), torch.empty((1))
