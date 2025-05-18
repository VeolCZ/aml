# import torch
# import joblib
# from numpy.typing import NDArray
# import numpy as np
# from abc import abstractmethod
# from matplotlib import pyplot as plt
# from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, StratifiedKFold
# import time
# from torch.utils.data import DataLoader
# from interface.ModelInterface import ModelInterface
# from preprocessing.ViTImageDataset import LabelType
# from preprocessing.TreeImageDataset import TreeImageDataset
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# class RandomForest(ModelInterface):
#     """
#     Base class for Random Forest models using scikit-learn.
#     This class is designed to be subclassed for specific tasks such as classification or regression.
#     """

#     def __init__(self) -> None:
#         """
#         Initializes the RandomForest instance.
#         Sets up the model and data containers.
#         Args:
#             cross_val(bool): Wheter crossvalidation will be applied.
#         """
#         self.x: list[NDArray] = []
#         self.y: list[LabelType] = []
#         self.model = self._init_model()

#     @abstractmethod
#     def _init_model(self) -> RandomForestClassifier | RandomForestRegressor:
#         pass

#     @abstractmethod
#     def _compute_metrics(self, y_test: list[torch.Tensor], y_pred: list[torch.Tensor]) -> float:
#         pass

#     @abstractmethod
#     def _get_target_key(self) -> str:
#         pass

#     def _collate(self, batch: list[tuple[torch.Tensor, dict]]) -> tuple[list[torch.Tensor], list[dict]]:
#         """
#         Collates a batch of data into a format suitable for the model.
#         Args:
#             batch (list[tuple[torch.Tensor, dict]]): A batch of data.
#         Returns:
#             tuple[list[torch.Tensor], list[dict]]: A tuple containing the images and labels."""
#         images, labels = zip(*batch)
#         return list(images), list(labels)

#     def _load_data(self) -> None:
#         """
#         Loads the training data from the TreeImageDataset.
#         This method uses the DataLoader to load the data in batches.
#         """
#         train_dataloader = DataLoader(TreeImageDataset("train"), collate_fn=self._collate)
#         idx = 0
#         for x_batch, y_batch in train_dataloader:
#             idx += 1
#             if idx >= 70:
#                 print("Karolina's computer likes to live")
#                 break
#             self.x.extend(x_batch)
#             self.y.extend(y_batch)
#         x_train, x_test, y_train, y_test = train_test_split(self.x,
#                                                             [label[target] for label in self.y],
#                                                             test_size=test_size,
#                                                             stratify=[label["cls"] for label in self.y],
#                                                             )

#     def fit(self, train_dataset: Dataset) -> float:
#         """
#         Fits the Random Forest model to the training data.
#         Args:
#             test_size (float): The proportion of the dataset to include in the test split.
#         Returns:
#             float: The score of the model on the test data.
#         """
#         target = self._get_target_key()
#         start_fit_time = time.perf_counter()

#         y_train = [y.squeeze() for y in y_train]
#         y_test = [y.squeeze() for y in y_test]
#         self.model.fit(x_train, y_train)
#         start_predict_time = time.perf_counter()
#         print(f"Model fitted in {start_predict_time - start_fit_time:.2f} seconds")
#         y_pred = self.predict(x_test)
#         return self._compute_metrics(y_test, y_pred)

#     def predict(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
#         """
#         Predicts the labels for the given input data.
#         Args:
#             x (list[torch.Tensor]): The input data.
#         Returns:
#             list[torch.Tensor]: The predicted labels."""
#         print("Predicting, this will take a while...")
#         preds = self.model.predict(x)
#         return [torch.tensor(p) for p in preds]

#     def cross_validation(self):
#         """
#         Performs cross-validation on the dataset.
#         Args:
#             n_splits (int): The number of splits for cross-validation.
#         """
#         target = self._get_target_key()
#         scores = []
#         start_c_time = time.perf_counter()
#         kfold = RepeatedStratifiedKFold(n_splits=2, n_repeats=2, random_state=42)
#         for train_index, test_index in kfold.split(self.x, [torch.argmax(label["cls"]).item() for label in self.y]):
#             x_train, x_test = [self.x[i] for i in train_index], [self.x[i] for i in test_index]
#             y_train, y_test = [self.y[i][target] for i in train_index], [self.y[i][target] for i in test_index]
#             # Perform training and evaluation here
#             self.model.fit(x_train, [y.squeeze() for y in y_train])
#             y_pred = self.predict(x_test)
#             score = self._compute_metrics([y.squeeze() for y in y_test], y_pred)
#             scores.append(score)
#         avg_score = sum(scores) / len(scores)
#         end_c_time = time.perf_counter()
#         print(f"Cross validation done in {end_c_time - start_c_time:.2f}s")
#         print(f"Average Cross-validation score is: {avg_score}")
#         return score

#     def plot_learning_curve(self, steps: int = 4, cv_folds: int = 3) -> None:
#         target = self._get_target_key()
#         x = self.x
#         y_all = [label[target].squeeze() for label in self.y]
#         y_cls = [torch.argmax(label["cls"]).item() for label in self.y]

#         train_sizes = np.linspace(0.1, 1.0, steps)
#         train_scores = []
#         val_scores = []

#         stratkfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
#         for i in train_sizes:
#             train_size = int(len(x) * i)
#             x_i = x[:train_size]
#             y_i = y_all[:train_size]
#             y_cls_i = y_cls[:train_size]
#             train_scores_fold = []
#             val_scores_fold = []

#             for train_idx, val_idx in stratkfold.split(x_i, y_cls_i):
#                 x_train, x_val = [x_i[i] for i in train_idx], [x_i[i] for i in val_idx]
#                 y_train, y_val = [y_i[i] for i in train_idx], [y_i[i] for i in val_idx]

#                 self.model.fit(x_train, y_train)
#                 train_score = self.evaluate(x_train, y_train)
#                 val_score = self.evaluate(x_val, y_val)

#                 train_scores_fold.append(train_score)
#                 val_scores_fold.append(val_score)

#             train_scores.append(np.mean(train_scores_fold))
#             val_scores.append(np.mean(val_scores_fold))

#         plt.plot(train_sizes, train_scores, label='Training Score')
#         plt.plot(train_sizes, val_scores, label='Validation Score')
#         plt.xlabel('Training Set Size')
#         if self.model.__class__.__name__ == "RandomForestClassifier":
#             plt.ylabel('Mean Accuracy')
#         elif self.model.__class__.__name__ == "RandomForestRegressor":
#             plt.ylabel('Mean IOU')
#         plt.title('Learning Curve for Random Forest')
#         plt.legend()
#         plt.grid()
#         plt.savefig("learning_curve.png")
#         plt.show()
#         print("Learning curve plotted.")

#     def save_model(self, path: str) -> None:
#         """
#         Saves the trained model to the specified path.
#         Args:
#             path (str): The path to save the model.
#         """
#         joblib.dump(self.model, path)
#         print(f"Model saved to {path}")

#     def load_model(self, path: str) -> None:
#         """
#         Loads the model from the specified path.
#         Args:
#             path (str): The path to load the model from.
#         """
#         self.model = joblib.load(path)
#         print(f"Model loaded from {path}")

#     def train_forest(self, cross_validation: bool = True) -> None:
#         """
#         Trains the Random Forest model on the training data.
#         This method loads the data, fits the model, and saves it to a file.
#         """
#         print("Loading data...")
#         self._load_data()
#         print("Data loaded.")
#         if cross_validation:
#             score = self.cross_validation()
#         else:
#             score = self.fit(printing=True)
#         print("Training complete. Score:", score)

#     def evaluate(self, x: list[torch.Tensor], y_ground: list[torch.Tensor]) -> float:
#         """
#         Evaluates the model on the given input data.
#         Args:
#             x (list[torch.Tensor]): The input data.
#             y_ground (list[torch.Tensor]): The ground truth labels.
#         Returns:
#             float: The accuracy score of the model on the input data.
#         """
#         y = self.predict(x)
#         return self._compute_metrics(y, y_ground)

#     def evaluate_forest(self, path: str) -> None:
#         """
#         Evaluates the Random Forest model on the test data.
#         Args:
#             path (str): The path to load the model from.
#         """
#         self.load_model(path)
#         self._load_data()
#         x = [torch.tensor(sample) for sample in self.x]
#         label_key = self._get_target_key()
#         y = [label[label_key].squeeze() for label in self.y]
#         score = self.evaluate(x, y)
#         print("Evaluation score:", score)
