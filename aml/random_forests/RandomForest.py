import torch
import joblib
from typing import Literal
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from numpy.typing import NDArray
from preprocessing.ViTImageDataset import LabelType
from preprocessing.TreeImageDataset import TreeImageDataset
from torch.utils.data import DataLoader

TaskType = Literal["classification", "regression"]


class RandomForest:
    """
    A class that implements a Random Forest model for either classification or regression tasks.
    It uses the scikit-learn library for the underlying implementation.
    Attributes:
        task_type (str): The type of task, either 'classification' or 'regression'.
        model: The Random Forest model from scikit-learn.
    Methods:
        _compute_iou: Computes the Intersection over Union (IoU) of two bounding boxes.
        _compute_many_iou: Computes the average IoU of two lists of bounding boxes.
        _compute_metrics: Computes the evaluation metric based on the task type.
        _collate: Collates a batch of data into images and labels.
        _load_data: Loads the training data from the TreeImageDataset.
        train: Trains the Random Forest model on the training data.
        save_weights: Saves the trained model weights to a file.
        train_forest: Loads the data and trains the model.
    """

    def __init__(self, task_type: TaskType):
        self.task_type = task_type
        self.x: list[NDArray] = []
        self.y: list[LabelType] = []

        if task_type.lower() == "classification":
            self.model = RandomForestClassifier()
        elif task_type.lower() == "regression":
            self.model = RandomForestRegressor()
        else:
            raise ValueError(
                f"Unsupported task_type type {task_type}. Should be 'classification' or 'regression'."
            )

    def _compute_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> float:
        """
        Compute the Intersection over Union (IoU) of two bounding boxes.
        Args:
            box1: The first bounding box (test).
            box2: The second bounding box (prediction).
        Returns:
            The IoU of the two bounding boxes.
        """
        # box: [x1, y1, x2, y2]
        x1 = torch.max(torch.tensor([box1[0], box2[0]]))
        y1 = torch.max(torch.tensor([box1[1], box2[1]]))
        x2 = torch.min(torch.tensor([box1[2], box2[2]]))
        y2 = torch.min(torch.tensor([box1[3], box2[3]]))

        inter_area = torch.max(torch.tensor(
            0), x2 - x1) * torch.max(torch.tensor(0), y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        if union_area > 0:
            iou_tensor = inter_area / union_area
            return float(iou_tensor.item())
        else:
            return 0

    def _compute_many_iou(self, boxes1: list[torch.Tensor], boxes2: list[torch.Tensor]) -> float:
        """
        Compute the average Intersection over Union (IoU) of two lists of bounding boxes.
        Args:
            many_boxes1: The first list of bounding boxes (test boxes).
            many_boxes2: The second list of bounding boxes (predicted boxes).
        Returns:
            The average IoU of the two lists of bounding boxes.
        """
        i = 0
        sum = 0.0
        assert len(boxes1) == len(
            boxes2), "The two lists must have the same length"
        for i in range(len(boxes1)):
            sum += self._compute_iou(boxes1[i], boxes2[i])
            i += 1
        return sum / i

    def _compute_metrics(self, y_test: list[torch.Tensor], y_pred: list[torch.Tensor]) -> float:
        """
        Compute the evaluation metric based on the task type.
        Args:
            y_test: The true labels.
            y_pred: The predicted labels.
        Returns:
            The evaluation metric.
        """
        if self.task_type == "regression":
            return self._compute_many_iou(y_test, y_pred)
        elif self.task_type == "classification":
            print(y_pred[0])
            return float(accuracy_score(y_test, y_pred))

    def _collate(self, batch: list[tuple[torch.Tensor, dict]]) -> tuple[list[torch.Tensor], list[dict]]:
        """
        Custom collate function to handle the batch of data.
        Args:
            batch: A list of tuples containing the image tensor and the corresponding label dictionary.
        Returns:
            A tuple containing a list of image tensors and a list of label dictionaries.
        """
        images, labels = zip(*batch)
        return list(images), list(labels)

    def _load_data(self) -> None:
        """
        Load the training data from the TreeImageDataset.
        """
        train_dataloader = DataLoader(
            TreeImageDataset("train"), collate_fn=self._collate
        )
        for x_batch, y_batch in train_dataloader:
            self.x.extend(x_batch)
            self.y.extend(y_batch)

    def fit(self, test_size: float = 0.2) -> float:
        """
        Fit the Random Forest model to the training data.
        Args:
            test_size: The proportion of the dataset to include in the test split.
        Returns:
            The evaluation metric.
        """
        target = None
        if self.task_type == "regression":
            target = "bbox"
        elif self.task_type == "classification":
            target = "cls"
        x = self.x
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            [label[target] for label in self.y],
            test_size=test_size,
            stratify=[label["cls"] for label in self.y],
        )

        print("Fitting Model, this will take a while...")
        y_train = [y.squeeze() for y in y_train]
        y_test = [y.squeeze() for y in y_test]
        self.model.fit(x_train, y_train)
        y_pred = self.predict(x_test)
        eval_metric = self._compute_metrics(y_test, y_pred)
        return eval_metric

    def predict(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Predict on given data

        Args:
            x: input data

        Returns:
            predictions
        """
        print("Predicting, this will take a while...")
        preditions = self.model.predict(x)
        return [torch.tensor(pred) for pred in preditions]

    def evaluate(self, x: list[torch.Tensor], y_ground: list[torch.Tensor]) -> float:
        """
        Evaluate the model's predictions on a given input
        and ground truth.

        Args:
            x: the input
            y_ground: the ground truth

        Returns:
            metric's score
        """
        y = self.predict(x)
        return self._compute_metrics(y, y_ground)

    def save_model(self, path: str) -> None:
        """
        Save the trained model weights to a file.

        Args:
            path: path to save
        """
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """
        Loads the moadel from a given path.

        Args:
            path: the path to load the model
        """
        self.model = joblib.load(path)
        print(f"Model loaded from {path}")

    def train_forest(self) -> None:
        """
        Loading the data and training the model
        """
        print(f"Loading data for {self.task_type}...")
        self._load_data()
        print("Data loaded")
        fitness = self.fit()
        print("Model fit:", fitness)
        print("Tree trained")

    def evaluate_forest(self, path: str) -> None:
        """
        Loads the model and evaluates it on the data
        """
        self.load_model(path)
        self._load_data()

        x = list(map(lambda x: torch.tensor(x), self.x))

        label = "bbox" if self.task_type == "regression" else "cls"
        model_eval = self.evaluate(x, [y[label].squeeze() for y in self.y])

        print("Evaluation:", model_eval)


def train_classifier_forest() -> None:
    """
    Run the random forest training after instantiating the forest object.
    """
    PATH = "/logs/forest_classifier.pkl"
    forest = RandomForest(task_type="classification")
    forest.train_forest()
    forest.save_model(PATH)
    forest.evaluate_forest(PATH)


def train_regressor_forest() -> None:
    """
    Run the random forest training after instantiating the forest object.
    """
    PATH = "/logs/forest_regressor.pkl"
    forest = RandomForest(task_type="regression")
    forest.train_forest()
    forest.save_model(PATH)
    forest.evaluate_forest(PATH)
