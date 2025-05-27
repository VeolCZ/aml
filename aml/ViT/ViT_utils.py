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
from evaluator.Evaluator import Evaluator
from preprocessing.ViTImageDataset import ViTImageDataset
from tboard.summarywriter import write_summary
from tboard.plotting import plot_confusion_matrix

SEED = int(os.getenv("SEED", 123))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
TEST_SIZE = float(os.getenv("TEST_SIZE", 0.1))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_vit() -> None:
    """
    Trains a Vision Transformer (ViT) model on an image dataset.

    Global variables:
        device (torch.device): The device (CPU or CUDA) on which to perform computations.
    """

    # Datasets
    train_dataset = ViTImageDataset(type="train")

    all_labels = train_dataset.get_cls_labels()
    train_indices, _, _, _ = train_test_split(
        np.arange(len(train_dataset)),
        all_labels,
        test_size=TEST_SIZE,
        stratify=all_labels,
        random_state=SEED
    )

    train_dataset_subset = Subset(train_dataset, train_indices)

    # Train the model
    model = ViT()
    model.fit(train_dataset_subset)


def eval_vit() -> None:
    """
    Evaluates the ViT on several metrics including:
        Accuracy
        top 5 accuracy
        F1 score
        Multiroc
        Confusion matrix
    It prints these evaluations in the terminal.
    """
    model = ViT()
    model.load("/weights/ViT_2025-05-16-12-28.294240ValLoss_1.84.pth")
    model.to(device=DEVICE)

    eval_dataset = ViTImageDataset(type="eval")

    all_labels = eval_dataset.get_cls_labels()
    _, test_indices, _, _ = train_test_split(
        np.arange(len(eval_dataset)),
        all_labels,
        test_size=TEST_SIZE,
        stratify=all_labels,
        random_state=SEED
    )

    test_dataset = Subset(eval_dataset, test_indices)
    dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=multiprocessing.cpu_count(),
        pin_memory=True
    )

    x_all_batches: list[torch.Tensor] = []
    y_all_batches: list[torch.Tensor] = []
    z_all_batches: list[torch.Tensor] = []

    for _, (imgs_batch, labels_batch) in enumerate(dataloader):
        x_all_batches.append(imgs_batch)
        y_all_batches.append(labels_batch["cls"])
        z_all_batches.append(labels_batch["bbox"])

    x = torch.cat(x_all_batches, dim=0)
    y = torch.cat(y_all_batches, dim=0)
    z = torch.cat(z_all_batches, dim=0).squeeze(1)

    eval_res = Evaluator.eval(model, x, y, z)
    confusion_matrix = eval_res.confusion_matrix
    image = plot_confusion_matrix(confusion_matrix.cpu().numpy())

    writer = write_summary(run_name="ViT")
    writer.add_scalar("ViT/Accuracy", eval_res.accuracy, 0)
    writer.add_scalar("ViT/F1", eval_res.f1_score, 0)
    writer.add_scalar("ViT/top_k", eval_res.top_3, 0)
    writer.add_scalar("ViT/top_k", eval_res.top_5, 0)
    writer.add_scalar("ViT/multiroc", eval_res.multiroc, 0)
    writer.add_image("ViT/Confusion Matrix", image, 0)
    writer.add_scalar("ViT/IOU", eval_res.iou, 0)
    # add training plot here
    writer.close()

    logger = logging.getLogger("Forest Eval")
    logger.info(eval_res)


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
        number_of_folds = trial.suggest_int("n_of_folds", 8, 20, step=2)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
        annealing_rate = trial.suggest_float("annealing_rate", 1e-8, learning_rate, log=True)

        model = ViT()
        trainer = ViTTrainer(model, device=DEVICE, dataset=ViTImageDataset(type="train"),
                             learning_rate=learning_rate, n_splits=number_of_folds, epochs=20,
                             batch_size=BATCH_SIZE,
                             patience=3, annealing_rate=annealing_rate)
        return float(trainer.train())

    STUDY_NAME = "vit_hyperparams"
    STORAGE_URL = f"sqlite:////logs/{STUDY_NAME}.db"

    study = optuna.create_study(direction="minimize", load_if_exists=True, study_name=STUDY_NAME,
                                storage=STORAGE_URL,)
    study.optimize(objective, n_trials=trial_count,
                   show_progress_bar=True, gc_after_trial=True, n_jobs=1)

    logger = logging.getLogger("HyperparameterOptimizer")
    study_df = study.trials_dataframe()
    study_df.to_csv("/logs/STUDY_NAME.csv", index=False)
    logger.info(study.best_params)

    return study.best_params
