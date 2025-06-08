import os
import optuna
import logging
import torch
import multiprocessing
import gc
from ViT.ViT import ViT
from torch.utils.data import DataLoader
from util import set_seeds
from ViT.ViTTrainer import ViTTrainer
from preprocessing.data_util import get_data_splits
from evaluator.Evaluator import Evaluator
from preprocessing.ViTImageDataset import ViTImageDataset, robustness_type
from preprocessing.data_util import load_data_to_mem

SEED = int(os.getenv("SEED", "123"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.1"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_vit(n: int = 5) -> None:
    """
    Trains one or more Vision Transformer (ViT) models.

    Args:
        n (int): The number of models to train with different seeds.
    """
    assert os.path.exists("/data/CUB_200_2011"), "Please ensure the dataset is properly extracted into /data"
    assert os.path.exists("/logs"), "Please ensure the /logs directory exists"
    assert os.path.exists("/weights"), "Please ensure the /weights directory exists"
    assert os.path.exists("/data/labels.csv"), "Please ensure the labels are generated (--make_labels)"

    if not os.path.exists("/weights/ViT"):
        os.makedirs("/weights/ViT", exist_ok=True)
    logger = logging.getLogger("ViT Trainer")

    for i in range(n):
        iter_seed = SEED + i
        set_seeds(iter_seed)
        logger.info(f"Training with seed {iter_seed}")

        model = ViT()
        train_dataset, val_dataset, _ = get_data_splits(
            ViTImageDataset("train"), ViTImageDataset("eval"), seed=iter_seed
        )

        model.fit(train_dataset, val_dataset)
        model.save(f"/weights/ViT/{iter_seed}")

        del model, train_dataset, val_dataset
        gc.collect()
        torch.cuda.empty_cache()


def eval_vit(n: int = 5) -> None:
    """
    Evaluates pre-trained Vision Transformer (ViT) models.

    Args:
        n (int): The number of saved models to evaluate.
    """
    assert os.path.exists("/data/CUB_200_2011"), "Please ensure the dataset is properly extracted into /data"
    assert os.path.exists("/logs"), "Please ensure the /logs directory exists"
    assert os.path.exists("/weights/ViT"), "Please ensure the /weights/ViT directory exists"
    assert os.path.exists("/data/labels.csv"), "Please ensure the labels are generated (--make_labels)"

    logger = logging.getLogger("ViT Eval")
    for i in range(n):
        iter_seed = SEED + i
        set_seeds(iter_seed)
        logger.info(f"Evaluating with seed {iter_seed}")
        model = ViT()
        model.load(f"/weights/ViT/{iter_seed}.pth")

        _, _, test_dataset = get_data_splits(ViTImageDataset("train"), ViTImageDataset("eval"), seed=iter_seed)
        x, y, z = load_data_to_mem(test_dataset)

        eval_res = Evaluator.eval(model, x, y, z, tag=f"ViT_eval_s{iter_seed}")
        logger.info(eval_res)


def optimize_hyperparameters(trial_count: int = 10) -> dict[str, float]:
    """
    Optimizes hyperparameters for the ViT model using Optuna.

    This function performs a hyperparameter search to find the optimal
    parameters for the ViTTrainer. Results of the study can be found in /logs.

    Args:
        trial_count (int, optional): The number of optimization trials to run.
                                     Defaults to 10.

    Returns:
        dict[str, float | int]: A dictionary containing the best hyperparameters
                                found by Optuna.
    """

    assert os.path.exists("/data/CUB_200_2011"), "Please ensure the dataset is properly extracted into /data"
    assert os.path.exists("/logs"), "Please ensure the /logs directory exists"
    assert os.path.exists("/data/labels.csv"), "Please ensure the labels are generated (--make_labels)"

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
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
        annealing_rate = trial.suggest_float("annealing_rate", 1e-10, learning_rate, log=True)
        dropout_rate = trial.suggest_float("dropout_rate", 0.01, 0.1, step=0.01)
        train_dataset, val_dataset, _ = get_data_splits(ViTImageDataset("train"), ViTImageDataset("eval"), seed=SEED)
        model = ViT(dp_rate=dropout_rate)
        trainer = ViTTrainer(
            model,
            device=DEVICE,
            learning_rate=learning_rate,
            epochs=20,
            batch_size=BATCH_SIZE,
            patience=2,
            annealing_rate=annealing_rate,
        )

        traind_dl = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=multiprocessing.cpu_count(),
            pin_memory=DEVICE.type == "cuda",
        )
        val_dl = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=multiprocessing.cpu_count(),
            pin_memory=DEVICE.type == "cuda",
        )
        return float(trainer.train(traind_dl, val_dl))

    STUDY_NAME = "vit_hyperparams"
    STORAGE_URL = f"sqlite:////logs/{STUDY_NAME}.db"

    study = optuna.create_study(
        direction="minimize",
        load_if_exists=True,
        study_name=STUDY_NAME,
        storage=STORAGE_URL,
    )
    study.optimize(objective, n_trials=trial_count, show_progress_bar=True, gc_after_trial=True, n_jobs=1)

    logger = logging.getLogger("HyperparameterOptimizer")
    study_df = study.trials_dataframe()
    study_df.to_csv(f"/logs/{STUDY_NAME}.csv", index=False)
    logger.info(study.best_params)

    return study.best_params


def get_one_robustness_evaluation(noise_severity: float, alteration_type: robustness_type, n: int = 5) -> None:
    assert os.path.exists("/data/CUB_200_2011"), "Please ensure the dataset is properly extracted into /data"
    assert os.path.exists("/logs"), "Please ensure the /logs directory exists"
    assert os.path.exists("/weights/ViT"), "Please ensure the /weights/ViT directory exists"
    assert os.path.exists("/data/labels.csv"), "Please ensure the labels are generated (--make_labels)"

    logger = logging.getLogger("ViT Eval")
    for i in range(n):
        iter_seed = SEED + i
        set_seeds(iter_seed)
        logger.info(f"Evaluating with seed {iter_seed}")
        model = ViT()
        model.load(f"/weights/ViT/{iter_seed}.pth")

        _, _, test_dataset = get_data_splits(
            ViTImageDataset("train"),
            ViTImageDataset(type="robustness", noise_severity=noise_severity, alteration_type=alteration_type),
            seed=iter_seed,
        )
        x, y, z = load_data_to_mem(test_dataset)

        eval_res = Evaluator.eval(model, x, y, z, tag=f"ViT_robustness_s{iter_seed}")
        logger.info(eval_res)


def calculate_robustness(distortion_type: robustness_type = "gaussian", severity_step: float = 0.1) -> None:
    severity = 0.0
    while severity <= 1:
        get_one_robustness_evaluation(severity, distortion_type)
        severity += severity_step
