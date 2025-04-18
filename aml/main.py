import sys

from hello_world import hello


def help_func() -> None:
    print("Available commands:")
    for arg, (_, desc) in command_map.items():
        print(f"  {arg}: {desc}")
    print("\nYou can combine multiple commands which) ", end="")
    print("will be executed in order, e.g.: --hello --goodbye")


command_map = {
    '--hello': (hello, "Hello world"),
    '--help': (help_func, "Shows this help message"),
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
