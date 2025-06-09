import logging
import os
import sys
from util import set_seeds
from ViT.ViT_utils import eval_vit, optimize_hyperparameters, train_vit, eval_vit_robustnes
from make_visualisations import make_visualization
from api.api import serve
from make_labels import make_labels
from random_forests.forest_utils import train_composite, eval_composite, eval_composite_robustnes
from streamlit_app.run_streamlit import run_streamlit

logging.basicConfig(
    level="INFO",
    filename="/logs/logs.log",  # Enable for preserved logs
    format="%(asctime)s %(levelname)s %(module)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

SEED = int(os.getenv("SEED", "123"))
set_seeds(SEED)


def help_func() -> None:
    """
    Prints a help message listing all available commands and their descriptions.
    """
    print("Available commands:")
    for arg, (_, desc) in COMMAND_MAP.items():
        print(f"  {arg}: {desc}")
    print("Example usage: bash run.sh --streamlit")


COMMAND_MAP = {
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
    "--robustness_vit": (eval_vit_robustnes, "Calculates the robustness of the ViT"),
    "--robustness_forest": (eval_composite_robustnes, "Calculates the robustness of the random forest")
}


def main() -> None:
    """
    Parses command-line arguments and executes the corresponding function.
    """
    if len(sys.argv) != 2 or all(arg not in COMMAND_MAP for arg in sys.argv[1:]):
        help_func()
    else:
        executed_commands = set()

        for arg in sys.argv[1:]:
            if arg in COMMAND_MAP and arg not in executed_commands:
                func, _ = COMMAND_MAP[arg]
                func()
                executed_commands.add(arg)
            elif arg not in COMMAND_MAP:
                print(f"Warning: Unknown command '{arg}'")


if __name__ == "__main__":
    main()
