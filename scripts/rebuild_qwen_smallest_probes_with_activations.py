#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.nb.experiments.base import ExperimentConfig
from src.nb.experiments.confidence_calibration import ConfidenceCalibrationExperiment
from src.nb.experiments.length import LengthBiasExperiment
from src.nb.experiments.position import PositionBiasExperiment
from src.nb.experiments.sycophancy import SycophancyBiasExperiment

logger = logging.getLogger(__name__)


RUN_SPECS = [
    (
        "length",
        PROJECT_ROOT / "experiments/configs/length_skywork_qwen-smallest_gsm8k.yaml",
        LengthBiasExperiment,
    ),
    (
        "position",
        PROJECT_ROOT / "experiments/configs/position_skywork_qwen-smallest_gsm8k.yaml",
        PositionBiasExperiment,
    ),
    (
        "sycophancy",
        PROJECT_ROOT / "experiments/configs/sycophancy_skywork_qwen-smallest_gsm8k_mc.yaml",
        SycophancyBiasExperiment,
    ),
    (
        "confidence",
        PROJECT_ROOT / "experiments/configs/confidence_calibration_ece_skywork_qwen-smallest_gsm8k_mc.yaml",
        ConfidenceCalibrationExperiment,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild Qwen-smallest probes and save probe activations.")
    parser.add_argument(
        "--only",
        choices=[name for name, _, _ in RUN_SPECS],
        action="append",
        help="Run only selected probe families.",
    )
    parser.add_argument("--artifacts-dir", default="artifacts_qwen_smallest_probe_rebuild", help="Output artifacts root.")
    parser.add_argument("--plots-dir", default="plots_qwen_smallest_probe_rebuild", help="Output plots root.")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    selected = set(args.only or [name for name, _, _ in RUN_SPECS])

    for name, config_path, exp_cls in RUN_SPECS:
        if name not in selected:
            continue

        config = ExperimentConfig.from_yaml(config_path)
        config.artifacts_dir = args.artifacts_dir
        config.plots_dir = args.plots_dir
        config.device = args.device
        config.extra = dict(config.extra)
        config.extra["save_probe_activations"] = True

        logger.info("Running %s from %s", name, config_path)
        exp_cls(config).run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    main()
