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
    assert os.path.exists(
        "/data/CUB_200_2011"
    ), "Please ensure the dataset is properly extracted into /data"
    assert os.path.exists("/logs"), "Please ensure the /logs directory exists"
    assert os.path.exists("/weights"), "Please ensure the /weights directory exists"
    assert os.path.exists(
        "/data/labels.csv"
    ), "Please ensure the labels are generated (--make_labels)"

    # Config
    SEED = int(os.getenv("SEED", 123))
    TEST_SIZE = 0.1
    torch.manual_seed(SEED)

    # Datasets
    train_dataset = ViTImageDataset(type="train")

    all_labels = train_dataset.get_cls_labels()
    train_indices, _, _, _ = train_test_split(
        np.arange(len(train_dataset)),
        all_labels,
        test_size=TEST_SIZE,
        stratify=all_labels,
        random_state=SEED,
    )

    train_dataset_subset = Subset(train_dataset, train_indices)

    # Train the model
    model = ViT()
    model.fit(train_dataset_subset)


def eval_vit() -> None:
    assert os.path.exists(
        "/data/CUB_200_2011"
    ), "Please ensure the dataset is properly extracted into /data"
    assert os.path.exists("/logs"), "Please ensure the /logs directory exists"
    assert os.path.exists("/weights"), "Please ensure the /weights directory exists"
    assert os.path.exists(
        "/data/labels.csv"
    ), "Please ensure the labels are generated (--make_labels)"
    assert os.path.exists(
        "/weights/ViT_2025-05-16_ValLoss_1.84.pth"
    ), "Please ensure that you have the latest weights"

    # Config
    SEED = int(os.getenv("SEED", 123))
    TEST_SIZE = 0.1
    torch.manual_seed(SEED)

    model = ViT()
    model.load("/weights/ViT_2025-05-16_ValLoss_1.84.pth")
    eval_dataset = ViTImageDataset(type="eval")

    all_labels = eval_dataset.get_cls_labels()
    _, test_indices, _, _ = train_test_split(
        np.arange(len(eval_dataset)),
        all_labels,
        test_size=TEST_SIZE,
        stratify=all_labels,
        random_state=SEED,
    )
    _test_dataset = Subset(eval_dataset, test_indices)


def optimize_hyperparameters(trial_count: int = 30) -> dict[str, float]:
    """
    Optimizes hyperparameters for the ViT model using Optuna.

    This function performs a hyperparameter search to find the optimal
    parameters for the ViTTrainer. Results of the study can be found in /logs.

    Args:
        trial_count (int, optional): The number of optimization trials to run.
                                     Defaults to 30.

    Returns:
        dict[str, float | int]: A dictionary containing the best hyperparameters
                                found by Optuna.
    """

    assert os.path.exists(
        "/data/CUB_200_2011"
    ), "Please ensure the dataset is properly extracted into /data"
    assert os.path.exists("/logs"), "Please ensure the /logs directory exists"
    assert os.path.exists(
        "/data/labels.csv"
    ), "Please ensure the labels are generated (--make_labels)"

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
        annealing_rate = trial.suggest_float(
            "annealing_rate", 1e-8, learning_rate, log=True
        )

        model = ViT()
        trainer = ViTTrainer(
            model,
            device=device,
            dataset=ViTImageDataset(type="train"),
            learning_rate=learning_rate,
            n_splits=number_of_folds,
            epochs=20,
            batch_size=150,
            patience=3,
            annealing_rate=annealing_rate,
        )
        return float(trainer.train())

    STUDY_NAME = "vit_hyperparams"
    STORAGE_URL = f"sqlite:////logs/{STUDY_NAME}.db"

    study = optuna.create_study(
        direction="minimize",
        load_if_exists=True,
        study_name=STUDY_NAME,
        storage=STORAGE_URL,
    )
    study.optimize(
        objective,
        n_trials=trial_count,
        show_progress_bar=True,
        gc_after_trial=True,
        n_jobs=2,
    )

    logger = logging.getLogger("HyperparameterOptimizer")
    study_df = study.trials_dataframe()
    study_df.to_csv(f"/logs/{STUDY_NAME}.csv", index=False)
    logger.info(study.best_params)

    return study.best_params
