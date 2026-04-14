#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("SAERM_ROOT", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(PROJECT_ROOT))

from src.nb.experiments.base import ExperimentConfig
from src.nb.experiments.length import LengthBiasExperiment
from src.nb.experiments.position import PositionBiasExperiment
from src.nb.experiments.sycophancy import SycophancyBiasExperiment
from src.nb.experiments.uncertainty import UncertaintyBiasExperiment

logger = logging.getLogger(__name__)


RUN_SPECS = [
    # Allen RM
    ("length",      "experiments/configs/length_allen_gsm8k.yaml",       LengthBiasExperiment,      "allen"),
    ("position",    "experiments/configs/position_allen_gsm8k.yaml",      PositionBiasExperiment,    "allen"),
    ("sycophancy",  "experiments/configs/sycophancy_allen_gsm8k_mc.yaml", SycophancyBiasExperiment,  "allen"),
    ("uncertainty", "experiments/configs/uncertainty_allen_gsm8k_mc.yaml", UncertaintyBiasExperiment, "allen"),
    # DeBERTa RM
    ("length",      "experiments/configs/length_deberta_gsm8k.yaml",       LengthBiasExperiment,      "deberta"),
    ("position",    "experiments/configs/position_deberta_gsm8k.yaml",     PositionBiasExperiment,    "deberta"),
    ("sycophancy",  "experiments/configs/sycophancy_deberta_gsm8k_mc.yaml", SycophancyBiasExperiment, "deberta"),
    ("uncertainty", "experiments/configs/uncertainty_deberta_gsm8k_mc.yaml", UncertaintyBiasExperiment, "deberta"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild Allen and DeBERTa probes and save activations for INLP analysis."
    )
    parser.add_argument(
        "--only",
        choices=list({name for name, _, _, _ in RUN_SPECS}),
        action="append",
        help="Run only selected probe families.",
    )
    parser.add_argument(
        "--rm",
        choices=["allen", "deberta"],
        action="append",
        help="Run only selected reward models.",
    )
    parser.add_argument("--artifacts-dir", default="artifacts_inlp_rebuild", help="Output artifacts root.")
    parser.add_argument("--plots-dir", default="plots_inlp_rebuild", help="Output plots root.")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_families = set(args.only or [name for name, _, _, _ in RUN_SPECS])
    selected_rms = set(args.rm or ["allen", "deberta"])

    for name, config_path, exp_cls, rm in RUN_SPECS:
        if name not in selected_families or rm not in selected_rms:
            continue

        config_full = PROJECT_ROOT / config_path
        if not config_full.exists():
            logger.warning("Config not found, skipping: %s", config_full)
            continue

        config = ExperimentConfig.from_yaml(config_full)
        config.artifacts_dir = args.artifacts_dir
        config.plots_dir = args.plots_dir
        config.device = args.device
        config.extra = dict(config.extra)
        config.extra["save_probe_activations"] = True

        logger.info("Running %s/%s from %s", rm, name, config_path)
        exp_cls(config).run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    main()
