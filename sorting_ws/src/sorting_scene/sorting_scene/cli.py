"""Command-line tool: print a randomized scene as JSON.

ros2 run sorting_scene generate_scene --seed 7
"""

import argparse

from sorting_scene.generator import generate_scene
from sorting_scene.model import default_config_path, load_config


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and print the generated scene."""
    parser = argparse.ArgumentParser(description="Generate a reproducible random scene.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-items", type=int, default=None)
    parser.add_argument("--config", default=str(default_config_path()))
    args = parser.parse_args(argv)
    print(generate_scene(load_config(args.config), args.seed, args.n_items).to_json())


if __name__ == "__main__":
    main()
