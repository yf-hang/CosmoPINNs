"""CosmoPINNs entry point, using the clean config2.json by default."""

from __future__ import annotations

import argparse
from pathlib import Path

import main_core as _core
from main_core import Config


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().with_name("config2.json"),
        help="Training configuration (default: config2.json next to main.py).",
    )
    return parser.parse_args(argv)


def main(config_path=None):
    return _core.main(config_path=config_path)


if __name__ == "__main__":
    main(_parse_args().config)
