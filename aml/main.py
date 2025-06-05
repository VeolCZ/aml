import logging
import os
import random
import sys
import numpy as np
import torch
from ViT.ViT_utils import eval_vit, optimize_hyperparameters, train_vit, calculate_robustness
from make_visualisations import make_visualization
from api.api import serve
from make_labels import make_labels
from random_forests.forest_utils import train_composite, eval_composite
from streamlit_app.run_streamlit import run_streamlit


logging.basicConfig(
    level="INFO",
    # filename="/logs/logs.log",  # Enable for preserved logs
    format="%(asctime)s %(levelname)s %(module)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

SEED = int(os.getenv("SEED", "123"))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def help_func() -> None:
    print("Available commands:")
    for arg, (_, desc) in command_map.items():
        print(f"  {arg}: {desc}")
    print("Example usage: bash run.sh --streamlit")


command_map = {
    "--help": (help_func, "Shows this help message"),
    "--make_labels": (make_labels, "Creates a labels.csv file in /data"),
    "--make_visualization": (make_visualization, "Create visualizations of /data"),
    "--eval_vit": (eval_vit, "Eval ViT"),
    "--optimize_hyperparams": (optimize_hyperparameters, "Optimize hyperparameters for ViT"),
    "--train_vit": (train_vit, "Trains the ViT model"),
    "--train_forest": (train_composite, "Train random forests"),
    "--eval_forest": (eval_composite, "Eval random forests"),
    "--serve": (serve, "Serve the models through API"),
    "--streamlit": (run_streamlit, "Run the streamlit application"),
    "--robustness_vit": (calculate_robustness, "Calculates the robustness of the model"),
}


def main() -> None:
    if len(sys.argv) != 2 or all(arg not in command_map for arg in sys.argv[1:]):
        help_func()
    else:
        executed_commands = set()

        for arg in sys.argv[1:]:
            if arg in command_map and arg not in executed_commands:
                func, _ = command_map[arg]
                func()
                executed_commands.add(arg)
            elif arg not in command_map:
                print(f"Warning: Unknown command '{arg}'")


if __name__ == "__main__":
    main()
