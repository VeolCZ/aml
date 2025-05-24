import multiprocessing
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import torch
from random_forests.RandomForest import RandomForest
from torch.utils.data import Dataset, DataLoader
from joblib import parallel_backend


class RandomForestClassifierModel(RandomForest):
    """
    Random Forest model for classification tasks.
    Inherits from the abstract base RandomForest class.
    """

    def __init__(self) -> None:
        super().__init__(RandomForestClassifier(n_jobs=-1,
                                                random_state=123, n_estimators=1_000))  # ADD SEED

    def fit(self, train_dataset: Dataset) -> None:
        x_train, y_train = [], []
        dataloader = DataLoader(
            train_dataset,
            batch_size=320,
            shuffle=True,
            num_workers=multiprocessing.cpu_count(),
        )

        for _, (imgs, labels) in enumerate(dataloader):
            x_train.append(imgs)
            y_train.append(np.array([label for label in labels["cls"]]))

        x_train_ds = np.concatenate(x_train, axis=0)
        y_train_ds = np.concatenate(y_train, axis=0)

        with parallel_backend("loky", n_jobs=-1):
            self.model.fit(x_train_ds, y_train_ds)

    def predict(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cls = self.model.predict_proba(data)

        return torch.empty((1)), torch.tensor(cls)
