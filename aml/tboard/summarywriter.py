from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from torch import tensor
import datetime
import os
"""
Access eval metrics an somehow extract them from dict or another method
Graphs:
- matrix for forests
- matrix for trees
- loss plot for vit
- time for inference
maybe plots:
- accuracy with num of images
- iou with num of images
- f1 with num of images
- hyperparams for ViT (maybe)
- nice display of all eval metrics instead of the current implementation
"""


def write_summary(run_name: str = "random_forest", base_log_dir: str="runs")-> SummaryWriter:
    """
    Function for writing tensorboard
    Args:
        run_name(str): name of the run will be used as filename
        base_log_dir(str): where the summarywriter will be stored (default = "runs").
    Returns
        summarywriter(SummaryWriter): an instance of the SummaryWriter class.
    """
    timestamp = "thing"
    log_dir = os.path.join(base_log_dir, f"{run_name}_{timestamp}")
    return SummaryWriter(log_dir=log_dir)


