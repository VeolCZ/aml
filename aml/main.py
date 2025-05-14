import sys
from make_visualisations import make_visualization
from make_labels import make_labels
from random_forests.forest_train_util import train_classifier_forest, train_regressor_forest


def help_func() -> None:
    print("Available commands:")
    for arg, (_, desc) in command_map.items():
        print(f"  {arg}: {desc}")
    print("\nYou can combine multiple commands which will be executed in order, e.g.: --parse --goodbye")


command_map = {
    "--help": (help_func, "Shows this help message"),
    "--make_labels": (make_labels, "Creates a labels.csv file in /data"),
    "--make_visualization": (make_visualization, "Create visualizations of /data"),
    "--forest_regressor": (train_regressor_forest, "Run the random forest regressor training"),
    "--forest_classifier": (train_classifier_forest, "Run the random forest classifier training"),
}


def main() -> None:
    if len(sys.argv) < 2 or all(arg not in command_map for arg in sys.argv[1:]):
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
    train_classifier_forest()
