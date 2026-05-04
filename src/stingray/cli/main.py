from __future__ import annotations

import argparse
from collections.abc import Callable
from importlib import import_module


COMMANDS = {
    ("sensors", "merge"): {
        "help": "Merge sensor data",
        "target": "stingray.cli.sensors:main",
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
    ("ctd", "download"): {
        "help": "Download CTD cruise data",
        "target": "ctd_tools.download:main",
    }
}


def load_target(target: str) -> Callable[[list[str] | None], None]:
    module_name, func_name = target.split(":", 1)
    module = import_module(module_name)
    func = getattr(module, func_name)

    if not callable(func):
        raise TypeError(f"Target is not callable: {target}")

    return func


def run_target(target: str, argv: list[str]) -> None:
    func = load_target(target)
    func(argv)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stingray",
        description="Stingray command-line tools",
    )

    groups = parser.add_subparsers(
        dest="group",
        metavar="<group>",
        required=True,
    )

    group_parsers: dict[str, argparse._SubParsersAction] = {}

    for (group, command), spec in COMMANDS.items():
        if group not in group_parsers:
            group_parser = groups.add_parser(
                group,
                help=f"{group} commands",
            )
            group_parsers[group] = group_parser.add_subparsers(
                dest="command",
                metavar="<command>",
                required=True,
            )

        command_parser = group_parsers[group].add_parser(
            command,
            help=spec["help"],
            description=spec["help"],
        )
        command_parser.set_defaults(target=spec["target"])

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args, passthrough_args = parser.parse_known_args(argv)

    target = getattr(args, "target", None)
    if target is None:
        parser.error("No command selected")

    run_target(target, passthrough_args)


if __name__ == "__main__":
    main()