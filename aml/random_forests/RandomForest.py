import torch
import joblib
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from torch.utils.data import DataLoader
from preprocessing.ViTImageDataset import LabelType
from preprocessing.TreeImageDataset import TreeImageDataset
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class RandomForest(ABC):
    """
    Base class for Random Forest models using scikit-learn.
    This class is designed to be subclassed for specific tasks such as classification or regression.
    """

    def __init__(self) -> None:
        """
        Initializes the RandomForest instance.
        Sets up the model and data containers.
        """
        self.x: list[NDArray] = []
        self.y: list[LabelType] = []
        self.model = self._init_model()

    @abstractmethod
    def _init_model(self) -> RandomForestClassifier | RandomForestRegressor:
        pass

    @abstractmethod
    def _compute_metrics(self, y_test: list[torch.Tensor], y_pred: list[torch.Tensor]) -> float:
        pass

    @abstractmethod
    def _get_target_key(self) -> str:
        pass

    def _collate(self, batch: list[tuple[torch.Tensor, dict]]) -> tuple[list[torch.Tensor], list[dict]]:
        """
        Collates a batch of data into a format suitable for the model.
        Args:
            batch (list[tuple[torch.Tensor, dict]]): A batch of data.
        Returns:
            tuple[list[torch.Tensor], list[dict]]: A tuple containing the images and labels."""
        images, labels = zip(*batch)
        return list(images), list(labels)

    def _load_data(self) -> None:
        """
        Loads the training data from the TreeImageDataset.
        This method uses the DataLoader to load the data in batches.
        """
        train_dataloader = DataLoader(TreeImageDataset("train"), collate_fn=self._collate)
        idx = 0
        for x_batch, y_batch in train_dataloader:
            idx += 1
            if idx >= 50:
                print("Karolina's computer likes to live")
                break
            self.x.extend(x_batch)
            self.y.extend(y_batch)

    def cross_validation(self, n_splits: int):
        """
        Performs cross-validation on the dataset.
        Args:
            n_splits (int): The number of splits for cross-validation.
        """
        target = self._get_target_key()
        kfold = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=5, random_state=42)
        for train_index, test_index in kfold.split(self.x, [label[target] for label in self.y]):
            x_train, x_test = [self.x[i] for i in train_index], [self.x[i] for i in test_index]
            y_train, y_test = [self.y[i][target] for i in train_index], [self.y[i][target] for i in test_index]
            # Perform training and evaluation here
        return x_train, x_test, y_train, y_test

    def fit(self, test_size: float = 0.2, cross_validate: bool = True) -> float:
        """
        Fits the Random Forest model to the training data.
        Args:
            test_size (float): The proportion of the dataset to include in the test split.
            cross_validate(bool): Whether cross validation should be applied, the number of splits is
            1/test_size.
        Returns:
            float: The f1 score of the model on the test data.
        """
        target = self._get_target_key()
        if cross_validate:
            print("cross validation has begun")
            x_train, x_test, y_train, y_test = self.cross_validation(int(1/test_size))
        else:
            x_train, x_test, y_train, y_test = train_test_split(
                self.x,
            [label[target] for label in self.y],
            test_size=test_size,
            stratify=[label["cls"] for label in self.y],
            )
        print("Fitting Model, this will take a while...")
        print(y_train)
        print(y_test)
        y_train = [y.squeeze() for y in y_train]
        y_test = [y.squeeze() for y in y_test]
        self.model.fit(x_train, y_train)
        y_pred = self.predict(x_test)
        return self._compute_metrics(y_test, y_pred)

    def predict(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Predicts the labels for the given input data.
        Args:
            x (list[torch.Tensor]): The input data.
        Returns:
            list[torch.Tensor]: The predicted labels."""
        print("Predicting, this will take a while...")
        preds = self.model.predict(x)
        return [torch.tensor(p) for p in preds]

    def save_model(self, path: str) -> None:
        """
        Saves the trained model to the specified path.
        Args:
            path (str): The path to save the model.
        """
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """
        Loads the model from the specified path.
        Args:
            path (str): The path to load the model from.
        """
        self.model = joblib.load(path)
        print(f"Model loaded from {path}")

    def train_forest(self) -> None:
        """
        Trains the Random Forest model on the training data.
        This method loads the data, fits the model, and saves it to a file.
        """
        print("Loading data...")
        self._load_data()
        print("Data loaded.")
        score = self.fit()
        print("Training complete. Score:", score)

    def evaluate(self, x: list[torch.Tensor], y_ground: list[torch.Tensor]) -> float:
        """
        Evaluates the model on the given input data.
        Args:
            x (list[torch.Tensor]): The input data.
            y_ground (list[torch.Tensor]): The ground truth labels.
        Returns:
            float: The accuracy score of the model on the input data.
        """
        y = self.predict(x)
        return self._compute_metrics(y, y_ground)

    def evaluate_forest(self, path: str) -> None:
        """
        Evaluates the Random Forest model on the test data.
        Args:
            path (str): The path to load the model from.
        """
        self.load_model(path)
        self._load_data()
        x = [torch.tensor(sample) for sample in self.x]
        label_key = self._get_target_key()
        y = [label[label_key].squeeze() for label in self.y]
        score = self.evaluate(x, y)
        print("Evaluation score:", score)
