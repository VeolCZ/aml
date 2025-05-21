from sklearn.ensemble import RandomForestClassifier
import torch
from random_forests.RandomForestTest import RandomForestTest
from torch.utils.data import Dataset


class RandomForestClassifierModel(RandomForestTest):
    """
    Random Forest model for classification tasks.
    Inherits from the abstract base RandomForest class.
    """

    def __init__(self) -> None:
        super().__init__(RandomForestClassifier())

    def fit(self, train_dataset: Dataset) -> None:
        x_train, y_train = [], []
        for i in range(len(train_dataset)):
            img, label = train_dataset[i]
            x_train.append(img)
            y_train.append(label["cls"])

        self.model.fit(x_train, y_train)

    def predict(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cls = self.model.predict_proba(data)
        return torch.empty((1)), torch.tensor(cls).squeeze(-1)
