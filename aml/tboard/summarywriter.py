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


def write_summary(run_name: str = "random_forest", base_log_dir: str="runs"):
    timestamp = "thing"
    log_dir = os.path.join(base_log_dir, f"{run_name}_{timestamp}")
    return SummaryWriter(log_dir=log_dir)


