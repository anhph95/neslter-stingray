from __future__ import annotations

import argparse
from collections.abc import Callable
from importlib import import_module


COMMANDS = {
    ("sensors", "process"): {
        "help": "Merge/process sensor data",
        "target": "stingray.cli.sensors:main",
    },
    ("ctd", "download"): {
        "help": "Download CTD cruise data",
        "target": "stingray.getctd:main",
    },
    ("dashboard", "run"): {
        "help": "Run Dash dashboard",
        "target": "stingray.dashboard.app:main",
    },
    ("images", "abundance"): {
        "help": "Compute image abundance",
        "target": "stingray.images.abundance:cli_main",
    },
    ("images", "frame-timestamp"): {
        "help": "Build frame timestamp CSV",
        "target": "stingray.images.frame_timestamp:main",
    },
    ("images", "generate-training"): {
        "help": "Generate YOLO training data",
        "target": "stingray.images.generate_training:main",
    },
    ("images", "tator-links"): {
        "help": "Add Tator annotation links",
        "target": "stingray.images.get_tator_link:main",
    },
}


def load_target(target: str) -> Callable[[list[str]], None]:
    module_name, func_name = target.split(":", 1)
    module = import_module(module_name)
    return getattr(module, func_name)


def run_target(target: str, argv: list[str]) -> None:
    func = load_target(target)
    func(argv)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="stingray")
    groups = parser.add_subparsers(dest="command", required=True)

    group_parsers = {}

    for (group, command), spec in COMMANDS.items():
        if group not in group_parsers:
            group_parser = groups.add_parser(group, help=f"{group} commands")
            group_parsers[group] = group_parser.add_subparsers(
                dest=f"{group}_command",
                required=True,
            )

        command_parser = group_parsers[group].add_parser(
            command,
            help=spec["help"],
        )
        command_parser.set_defaults(target=spec["target"])

    args, unknown = parser.parse_known_args(argv)

    if not hasattr(args, "target"):
        parser.error("No command selected")

    run_target(args.target, unknown)


if __name__ == "__main__":
    main()