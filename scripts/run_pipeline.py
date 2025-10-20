#!/usr/bin/env python3
"""Run the full interpretable reward model pipeline."""

from __future__ import annotations

import argparse

from saerm.config import load_experiment_config
from saerm.logging import configure_logging
from saerm.pipeline import ExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Path to experiment config YAML")
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    args = parser.parse_args()

    configure_logging(args.log_level)
    config = load_experiment_config(args.config)
    runner = ExperimentRunner(config, log_level=args.log_level)
    runner.run_all()


if __name__ == "__main__":
    main()
