import logging
import os
import torch
import numpy as np
import multiprocessing
from ViT.ViT import ViT
from datetime import datetime
from ViT.ViTTrainer import ViTTrainer
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from preprocessing.ViTImageDataset import ViTImageDataset
import optuna

device = torch.device("cuda")


def train_vit() -> None:
    """
    Trains a Vision Transformer (ViT) model on an image dataset.

    This function performs the following steps:
    1.  Sets up the configuration for training, including seed, batch size, and model path.
    2.  Loads the training and evaluation datasets using `ViTImageDataset`.
    3.  Splits the training dataset into training and testing subsets.
    4.  Creates a DataLoader for the testing subset.
    5.  Initializes the ViT model and the `ViTTrainer`.
    6.  Trains the model using the `ViTTrainer`.

    The trained model artifacts are intended to be saved to a specified path,
    though the `save` parameter is currently set to `False` in the `trainer.train` call.

    Global variables:
        device (torch.device): The device (CPU or CUDA) on which to perform computations.
    """
    global device

    # Config
    SEED = int(os.getenv("SEED", 123))
    torch.manual_seed(SEED)
    batch_size = 350
    model_path = f"/data/ViT_{datetime.utcnow()}"

    # Datasets
    train_dataset = ViTImageDataset(type="train")
    eval_dataset = ViTImageDataset(type="eval")

    all_labels = train_dataset.get_cls_labels()

    train_indices, test_indices, _, _ = train_test_split(
        np.arange(len(train_dataset)),
        all_labels,
        test_size=0.1,
        stratify=all_labels,
        random_state=SEED
    )

    train_dataset_subset = Subset(train_dataset, train_indices)
    test_dataset = Subset(eval_dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=multiprocessing.cpu_count(), pin_memory=device.type == "cuda")

    # Train the model
    model = ViT()
    trainer = ViTTrainer(model, device, train_dataset_subset,
                         epochs=20, n_splits=2, batch_size=150)
    trainer.train(model_path=model_path, save=False)

    # Test the model
    # model.load_state_dict(torch.load("/data/ViT_2025-05-14 16:05:31.936127ValLoss_1.92.pth")).to(device)

    # total = 0
    # correct = 0
    # for images, labels in test_loader:
    # model.eval()
    # images = images.to(device)
    # bbox, cls = model(images)

    # predicted_class_indices = cls.argmax(-1)
    # actual_class_ids_batch = labels["cls"].argmax(-1).to(device)

    # for k_in_batch in range(len(predicted_class_indices)):
    # # assert k_in_batch < 128
    # pred_idx = predicted_class_indices[k_in_batch].item()
    # actual_cls_id = actual_class_ids_batch[k_in_batch].item()

    # if pred_idx == actual_cls_id:
    # correct += 1
    # total += 1
    # print(f"Predicted: {pred_idx}, Actual: {actual_cls_id}")
    # print(f"Accuracy {correct / total}")


def optimize_hyperparameters(trial_count: int = 2) -> dict[str, float]:
    """
    Optimizes hyperparameters for the ViT model using Optuna.

    This function performs a hyperparameter search to find the optimal
    learning rate and number of folds for k-fold cross-validation within
    the ViTTrainer. It uses Optuna's TPE (Tree-structured Parzen Estimator)
    sampler by default to explore the search space.

    The objective function trains a ViT model with a given set of
    hyperparameters suggested by Optuna and returns a metric (e.g., loss)
    that Optuna aims to minimize.

    Side Effects:
        - Logs the best parameters found during the study.
        - Saves a CSV file named "study.csv" in the "/logs/" directory
          (ensure this directory exists and is writable) containing details
          of all trials.

    Args:
        trial_count (int, optional): The number of optimization trials to run.
                                     Defaults to 100.

    Returns:
        dict[str, float | int]: A dictionary containing the best hyperparameters
                                found by Optuna. For example:
                                `{'Learning rate': 0.001, 'Number of folds': 7}`
    """
    def objective(trial: optuna.Trial) -> float:
        """
        Objective function for Optuna to minimize.

        This function takes an Optuna trial, suggests hyperparameters,
        initializes and trains a ViT model, and returns the value to be
        minimized (e.g., validation loss from the trainer).

        Args:
            trial (optuna.Trial): An Optuna trial object used to suggest
                                  hyperparameters.

        Returns:
            float: The objective value to minimize (e.g., loss).
                   The ViTTrainer.train() method is expected to return this.
        """
        global device

        number_of_folds = trial.suggest_int("n_of_folds", 5, 10, step=1)
        learning_rate = trial.suggest_float(
            "learning_rate", 1e-6, 1e-2, log=True)

        vit = ViT()
        trainer = ViTTrainer(vit, device=device, dataset=ViTImageDataset(type="train"),
                             lr=learning_rate, n_splits=number_of_folds)
        return trainer.train()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=trial_count,
                   show_progress_bar=True, gc_after_trial=True, n_jobs=2)

    logger = logging.getLogger("HyperparameterOptimizer")
    study_df = study.trials_dataframe()
    study_df.to_csv("/logs/study.csv", index=False)
    logger.info(study.best_params)

    return study.best_params
