import os
import optuna
import logging
import torch
import multiprocessing
from ViT.ViT import ViT
from torch.utils.data import DataLoader
from util import set_seeds
from ViT.ViTTrainer import ViTTrainer
from preprocessing.data_util import get_data_splits
from evaluator.Evaluator import Evaluator
from preprocessing.ViTImageDataset import ViTImageDataset
from tboard.summarywriter import write_summary
from tboard.plotting import plot_confusion_matrix

SEED = int(os.getenv("SEED", "123"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.1"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_vit(n: int = 2) -> None:
    assert os.path.exists("/data/CUB_200_2011"), "Please ensure the dataset is properly extracted into /data"
    assert os.path.exists("/logs"), "Please ensure the /logs directory exists"
    assert os.path.exists("/weights"), "Please ensure the /weights directory exists"
    assert os.path.exists("/data/labels.csv"), "Please ensure the labels are generated (--make_labels)"

    logger = logging.getLogger("ViT Trainer")
    for i in range(n):
        iter_seed = SEED+i
        set_seeds(iter_seed)
        logger.info(f"Training with seed {iter_seed}")

        model = ViT()
        train_dataset, val_dataset, _ = get_data_splits(ViTImageDataset("train"),
                                                        ViTImageDataset("eval"), seed=iter_seed)

        model.fit(train_dataset, val_dataset)
        model.save(f"/weights/ViT/{iter_seed}")


def eval_vit(n: int = 2) -> None:
    assert os.path.exists("/data/CUB_200_2011"), "Please ensure the dataset is properly extracted into /data"
    assert os.path.exists("/logs"), "Please ensure the /logs directory exists"
    assert os.path.exists("/weights/ViT"), "Please ensure the /weights/ViT directory exists"
    assert os.path.exists("/data/labels.csv"), "Please ensure the labels are generated (--make_labels)"

    logger = logging.getLogger("ViT Eval")
    for i in range(n):
        iter_seed = SEED+i
        set_seeds(iter_seed)
        logger.info(f"Evaluating with seed {iter_seed}")

        model = ViT()
        model.load(f"/weights/ViT/{iter_seed}.pth")

        _, _, test_dataset = get_data_splits(ViTImageDataset("train"),
                                             ViTImageDataset("eval"), seed=iter_seed)
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
        writer.add_scalar("ViT/Accuracy", eval_res.accuracy, iter_seed)
        writer.add_scalar("ViT/F1", eval_res.f1_score, iter_seed)
        writer.add_scalar("ViT/top_k", eval_res.top_3, iter_seed)
        writer.add_scalar("ViT/top_k", eval_res.top_5, iter_seed)
        writer.add_scalar("ViT/multiroc", eval_res.multiroc, iter_seed)
        writer.add_image("ViT/Confusion Matrix", image, iter_seed)
        writer.add_scalar("ViT/IOU", eval_res.iou, iter_seed)
        # add training plot here
        writer.close()

        logger.info(eval_res)


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
    study_df.to_csv(f"/logs/{STUDY_NAME}.csv", index=False)
    logger.info(study.best_params)

    return study.best_params
