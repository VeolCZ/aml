import os
import optuna
import logging
import torch
import multiprocessing
import gc
from ViT.ViT import ViT
from torch.utils.data import DataLoader
from tboard.summarywriter import write_summary
from util import set_seeds
from ViT.ViTTrainer import ViTTrainer
from preprocessing.data_util import get_data_splits
from evaluator.Evaluator import Evaluator
from preprocessing.ViTImageDataset import ViTImageDataset
from preprocessing.data_util import load_data_to_mem
from tboard.plotting import plot_confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from preprocessing.ViTPreprocessPipeline import ViTPreprocessPipeline

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

    logger = logging.getLogger("ViT Robustness Eval")
    for i in range(n):
        iter_seed = SEED + i
        set_seeds(iter_seed)
        logger.info(f"Evaluating with seed {iter_seed}")
        model = ViT()
        model.load(f"/weights/ViT/{iter_seed}.pth")

        _, _, test_dataset = get_data_splits(ViTImageDataset("train"), ViTImageDataset("eval"), seed=iter_seed)
        x, y, z = load_data_to_mem(test_dataset)

        tag = f"ViT_eval_s{iter_seed}"
        eval_res = Evaluator.eval(model, x, y, z)
        logger.info(eval_res)

        matrix_image = plot_confusion_matrix(eval_res.confusion_matrix.cpu().numpy())
        writer = write_summary(run_name=tag)
        writer.add_image("Confusion Matrix", matrix_image)
        hparams = {"seed": tag}

        metrics = {
            "accuracy": eval_res.accuracy,
            "f1_score": eval_res.f1_score,
            "auroc": eval_res.multiroc,
            "top_3_accuracy": eval_res.top_3,
            "top_5_accuracy": eval_res.top_5,
            "iou": eval_res.iou,
            "random_iou": eval_res.random_iou,
        }

        writer.add_hparams(hparams, metrics, run_name=tag)
        writer.add_scalar("accuracy", eval_res.accuracy)
        writer.close()


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


def get_one_robustness_evaluation(model: ViT, noise_severity: float, alteration_type: str,
                                  writer: SummaryWriter
                                  ) -> None:
    logger = logging.getLogger("ViT Eval")
    logger.info(f"Evaluating with seed {SEED}")

    robustness_dataset = ViTImageDataset("eval")
    robustness_dataset.set_transform(
        ViTPreprocessPipeline.get_base_robustness_transform(noise_severity, alteration_type))

    set_seeds(SEED)
    _, _, test_dataset = get_data_splits(
        ViTImageDataset("train"),
        robustness_dataset,
        seed=SEED,
    )
    x, y, z = load_data_to_mem(test_dataset)

    eval_res = Evaluator.eval(
        model,
        x,
        y,
        z,
    )

    writer.add_scalar(f"Accuracy/{alteration_type}", eval_res.accuracy, noise_severity*100)
    writer.add_scalar(f"IOU/{alteration_type}", eval_res.iou, noise_severity*100)


def eval_vit_robustnes() -> None:
    assert os.path.exists("/data/CUB_200_2011"), "Please ensure the dataset is properly extracted into /data"
    assert os.path.exists("/logs"), "Please ensure the /logs directory exists"
    assert os.path.exists("/weights/ViT"), "Please ensure the /weights/ViT directory exists"
    assert os.path.exists("/data/labels.csv"), "Please ensure the labels are generated (--make_labels)"

    model = ViT()
    model.load(f"/weights/ViT/{SEED}.pth")
    severity = 0.0
    severity_step = 0.05
    for type in ["gaussian", "saltandpepper", "motionblur", "superpixels"]:
        writer = write_summary(run_name=f"ViT_robustness_s{SEED}_{type}")

        severity = 0.0
        while severity <= 1:
            get_one_robustness_evaluation(model, severity, type, writer)
            severity += severity_step

        writer.close()
