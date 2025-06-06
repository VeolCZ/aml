import logging
import os
from util import set_seeds
from preprocessing.data_util import get_data_splits, load_data_to_mem
from random_forests.CompositeRandomForest import CompositeRandomForest
from evaluator.Evaluator import Evaluator
from preprocessing.TreeImageDataset import TreeImageDataset
from tboard.plotting import plot_confusion_matrix
from tboard.summarywriter import write_summary


SEED = int(os.getenv("SEED", "123"))


def train_composite(n: int = 1) -> None:
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


def eval_composite(n: int = 2) -> None:
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

        eval_res = Evaluator.eval(model, x, y, z)
        confusion_matrix = eval_res.confusion_matrix
        image = plot_confusion_matrix(confusion_matrix.cpu().numpy())

        writer = write_summary(run_name="Forest")
        writer.add_scalar("Classifier/Accuracy", eval_res.accuracy, iter_seed)
        writer.add_scalar("Classifier/F1", eval_res.f1_score, iter_seed)
        writer.add_scalar("Classifier/top_k", eval_res.top_3, iter_seed)
        writer.add_scalar("Classifier/top_k", eval_res.top_5, iter_seed)
        writer.add_scalar("Classifier/multiroc", eval_res.multiroc, iter_seed)
        writer.add_image("Classifier/Confusion Matrix", image, iter_seed)
        writer.add_scalar("Regressor/IOU", eval_res.iou, iter_seed)
        writer.close()

        logger.info(eval_res)
