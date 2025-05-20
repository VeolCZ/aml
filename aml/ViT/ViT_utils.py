import os
import optuna
import logging
import torch
import numpy as np
import multiprocessing
from ViT.ViT import ViT
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from ViT.ViTTrainer import ViTTrainer
from preprocessing.ViTImageDataset import ViTImageDataset


def train_vit() -> None:
    """
    Trains a Vision Transformer (ViT) model on an image dataset.

    Global variables:
        device (torch.device): The device (CPU or CUDA) on which to perform computations.
    """

    # Config
    device = torch.device("cuda")
    SEED = int(os.getenv("SEED", 123))
    torch.manual_seed(SEED)
    batch_size = 350

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
    model.to(device=device)
    model.fit(train_dataset_subset)


def eval_vit() -> None:
    # Config
    SEED = int(os.getenv("SEED", 123))
    torch.manual_seed(SEED)
    device = torch.device("cuda")

    model = ViT()
    model.to(device=device)

    eval_dataset = ViTImageDataset(type="eval")

    all_labels = eval_dataset.get_cls_labels()
    _, test_indices, _, _ = train_test_split(
        np.arange(len(eval_dataset)),
        all_labels,
        test_size=0.1,  # CHANGE
        stratify=all_labels,
        random_state=SEED
    )
    _test_dataset = Subset(eval_dataset, test_indices)


def optimize_hyperparameters(trial_count: int = 30) -> dict[str, float]:
    """
    Optimizes hyperparameters for the ViT model using Optuna.

    This function performs a hyperparameter search to find the optimal
    parameters for the ViTTrainer.

    Args:
        trial_count (int, optional): The number of optimization trials to run.
                                     Defaults to 30.

    Returns:
        dict[str, float | int]: A dictionary containing the best hyperparameters
                                found by Optuna.
    """
    def objective(trial: optuna.Trial) -> float:
        """
        Objective function for Optuna to minimize.

        Args:
            trial (optuna.Trial): An Optuna trial object used to suggest
                                  hyperparameters.

        Returns:
            float: The objective value to minimize (e.g., loss).
                   The ViTTrainer.train() method is expected to return this.
        """
        device = torch.device("cuda")

        number_of_folds = trial.suggest_int("n_of_folds", 8, 20, step=2)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
        annealing_rate = trial.suggest_float("annealing_rate", 1e-8, learning_rate, log=True)

        model = ViT()
        trainer = ViTTrainer(model, device=device, dataset=ViTImageDataset(type="train"),
                             learning_rate=learning_rate, n_splits=number_of_folds, epochs=20, batch_size=150,
                             patience=3, annealing_rate=annealing_rate)
        return float(trainer.train())

    STUDY_NAME = "vit_hyperparams"
    STORAGE_URL = f"sqlite:////logs/{STUDY_NAME}.db"

    study = optuna.create_study(direction="minimize", load_if_exists=True, study_name=STUDY_NAME,
                                storage=STORAGE_URL,)
    study.optimize(objective, n_trials=trial_count,
                   show_progress_bar=True, gc_after_trial=True, n_jobs=2)

    logger = logging.getLogger("HyperparameterOptimizer")
    study_df = study.trials_dataframe()
    study_df.to_csv("/logs/STUDY_NAME.csv", index=False)
    logger.info(study.best_params)

    return study.best_params
