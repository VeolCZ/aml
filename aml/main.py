import logging
import sys
from ViT.ViT_utils import train_vit
from make_visualisations import make_visualization
from make_labels import make_labels


logging.basicConfig(
    level="INFO",
    filename="/logs/log.txt",  # Enable for preserved logs
    format="%(asctime)s %(levelname)s %(module)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def help_func() -> None:
    print("Available commands:")
    for arg, (_, desc) in command_map.items():
        print(f"  {arg}: {desc}")
    print("\nYou can combine multiple commands which will be executed in order, e.g.: --parse --goodbye")


command_map = {
    "--make_labels": (make_labels, "Creates a labels.csv file in /data"),
    "--make_visualization": (make_visualization, "Create visualizations of /data"),
    "--help": (help_func, "Shows this help message"),
    "--test_vit": (train_vit, "tests vit")
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
    main()
