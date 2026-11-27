"""Run pipeline steps without Airflow.

uv run python -m pipelines.cli all
uv run python -m pipelines.cli validate_raw preprocess --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import logging

from pipelines.steps import EXTRA_STEPS, STEPS
from src.config import load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "steps",
        nargs="+",
        choices=[*STEPS, *EXTRA_STEPS, "all"],
        help="'all' runs the pipeline steps in order; decode_sweep and compare run on request",
    )
    parser.add_argument(
        "--config", default=None, help="YAML config (default: configs/default.yaml)"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    cfg = load_config(args.config)
    steps = list(STEPS) if "all" in args.steps else args.steps
    for name in steps:
        logging.info("=== %s ===", name)
        step = STEPS.get(name) or EXTRA_STEPS[name]
        print(json.dumps({name: step(cfg)}, indent=2, default=str))


if __name__ == "__main__":
    main()
