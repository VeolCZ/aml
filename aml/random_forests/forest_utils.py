import logging
import os
from util import set_seeds
from preprocessing.data_util import get_data_splits, load_data_to_mem
from random_forests.CompositeRandomForest import CompositeRandomForest
from evaluator.Evaluator import Evaluator
from preprocessing.TreeImageDataset import TreeImageDataset
from tboard.plotting import plot_confusion_matrix
from tboard.summarywriter import write_summary
from preprocessing.TreePrerocessPipeline import TreePrerocessPipeline
from torch.utils.tensorboard import SummaryWriter

SEED = int(os.getenv("SEED", "123"))


def train_composite(n: int = 5) -> None:
    """
    Trains one or more Composite Random Forest models.

    This function iterates n times, each time training a new model with a
    different random seed. Each trained model is saved to the
    /weights/forest directory, named by its seed.

    Args:
        n (int): The number of models to train with different seeds.
    """

    assert os.path.exists("/data/CUB_200_2011"), "Please ensure the dataset is properly extracted into /data"
    assert os.path.exists("/logs"), "Please ensure the /logs directory exists"
    assert os.path.exists("/weights"), "Please ensure the /weights directory exists"
    assert os.path.exists("/data/labels.csv"), "Please ensure the labels are generated (--make_labels)"

    logger = logging.getLogger("Forest Trainer")

    for i in range(n):
        iter_seed = SEED+i
        set_seeds(iter_seed)
        logger.info(f"Training with seed {iter_seed}")

        model = CompositeRandomForest()
        train_dataset, val_dataset, _ = get_data_splits(TreeImageDataset("train"),
                                                        TreeImageDataset("eval"), seed=iter_seed)

        model.fit(train_dataset, val_dataset)
        model.save(f"/weights/forest/{iter_seed}")


def eval_composite(n: int = 5) -> None:
    """
    Evaluates pre-trained Composite Random Forest models.

    For n specified models, this function loads each one, runs predictions
    on the test set, and logs performance metrics (e.g., accuracy, F1, IOU)
    and a confusion matrix to TensorBoard.

    Args:
        n (int): The number of saved models to evaluate.
    """
    assert os.path.exists("/data/CUB_200_2011"), "Please ensure the dataset is properly extracted into /data"
    assert os.path.exists("/logs"), "Please ensure the /logs directory exists"
    assert os.path.exists("/weights/forest"), "Please ensure the /weights/forest directory exists"
    assert os.path.exists("/data/labels.csv"), "Please ensure the labels are generated (--make_labels)"

    logger = logging.getLogger("Forest Eval")

    for i in range(n):
        iter_seed = SEED+i
        set_seeds(iter_seed)
        logger.info(f"Evaluating with seed {iter_seed}")

        model = CompositeRandomForest()
        model.load(f"/weights/forest/{iter_seed}")

        test_dataset: TreeImageDataset
        _, _, test_dataset = get_data_splits(TreeImageDataset("train"),
                                             TreeImageDataset("eval"), seed=iter_seed)
        x, y, z = load_data_to_mem(test_dataset)

        eval_res = Evaluator.eval(model, x, y, z, )
        logger.info(eval_res)

        tag = f"Forest_eval_s{iter_seed}"
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


def get_one_robustness_evaluation(model: CompositeRandomForest, noise_severity: float, alteration_type: str,
                                  writer: SummaryWriter
                                  ) -> None:
    logger = logging.getLogger("Forest Robustness Eval")
    logger.info(f"Evaluating with seed {SEED}")

    robustness_dataset = TreeImageDataset("eval")
    robustness_dataset.set_transform(
        TreePrerocessPipeline.get_base_robustness_transform(noise_severity, alteration_type))

    set_seeds(SEED)
    _, _, test_dataset = get_data_splits(
        TreeImageDataset("train"),
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


def eval_composite_robustnes() -> None:
    assert os.path.exists("/data/CUB_200_2011"), "Please ensure the dataset is properly extracted into /data"
    assert os.path.exists("/logs"), "Please ensure the /logs directory exists"
    assert os.path.exists("/weights/forest"), "Please ensure the /weights/forest directory exists"
    assert os.path.exists("/data/labels.csv"), "Please ensure the labels are generated (--make_labels)"

    model = CompositeRandomForest()
    model.load(f"/weights/forest/{SEED}")
    severity = 0.0
    severity_step = 0.05

    writer = write_summary(run_name=f"Forest_robustness_s{SEED}")
    for type in ["gaussian", "saltandpepper", "motionblur", "superpixels"]:
        severity = 0.0
        while severity <= 1:
            get_one_robustness_evaluation(model, severity, type, writer)
            severity += severity_step

    writer.close()
