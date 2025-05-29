from torch.utils.tensorboard import SummaryWriter
import os


def write_summary(run_name: str, base_log_dir: str = "/logs/tb") -> SummaryWriter:
    """
    Function for writing tensorboard
    Args:
        run_name(str): name of the run will be used as filename
        base_log_dir(str): where the summarywriter will be stored (default = "/logs").
    Returns
        summarywriter(SummaryWriter): an instance of the SummaryWriter class.
    """
    marker = "thing"
    log_dir = os.path.join(base_log_dir, f"{run_name}_{marker}")
    return SummaryWriter(log_dir=log_dir)
