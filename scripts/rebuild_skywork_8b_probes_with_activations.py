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
from src.nb.experiments.confidence_calibration import ConfidenceCalibrationExperiment
from src.nb.experiments.length import LengthBiasExperiment
from src.nb.experiments.position import PositionBiasExperiment
from src.nb.experiments.sycophancy import SycophancyBiasExperiment
from src.nb.experiments.uncertainty import UncertaintyBiasExperiment

logger = logging.getLogger(__name__)


RUN_SPECS = [
    (
        "length",
        "experiments/configs/length_skywork_gsm8k.yaml",
        LengthBiasExperiment,
    ),
    (
        "position",
        "experiments/configs/position_skywork_gsm8k.yaml",
        PositionBiasExperiment,
    ),
    (
        "sycophancy",
        "experiments/configs/sycophancy_skywork_gsm8k_mc.yaml",
        SycophancyBiasExperiment,
    ),
    (
        "uncertainty",
        "experiments/configs/uncertainty_skywork_gsm8k_mc.yaml",
        UncertaintyBiasExperiment,
    ),
    (
        "confidence",
        "experiments/configs/confidence_calibration_ece_skywork_qwen-smallest_gsm8k_mc.yaml",
        ConfidenceCalibrationExperiment,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild Skywork-8B probes and save probe activations for multiple bias families."
    )
    parser.add_argument(
        "--only",
        choices=[name for name, _, _ in RUN_SPECS],
        action="append",
        help="Run only selected probe families.",
    )
    parser.add_argument("--model-path", default="Skywork/Skywork-Reward-V2-Llama-3.1-8B")
    parser.add_argument("--project-root", default=None, help="Override SAERM project root.")
    parser.add_argument("--artifacts-dir", default="artifacts_skywork8b_probe_rebuild")
    parser.add_argument("--plots-dir", default="plots_skywork8b_probe_rebuild")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global PROJECT_ROOT
    if args.project_root:
        PROJECT_ROOT = Path(args.project_root).resolve()
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))

    selected = set(args.only or [name for name, _, _ in RUN_SPECS])

    for name, config_rel, exp_cls in RUN_SPECS:
        if name not in selected:
            continue

        config_path = PROJECT_ROOT / config_rel
        config = ExperimentConfig.from_yaml(config_path)
        config.model_path = args.model_path
        config.artifacts_dir = args.artifacts_dir
        config.plots_dir = args.plots_dir
        config.device = args.device
        config.extra = dict(config.extra)
        config.extra["save_probe_activations"] = True

        # Confidence config in repo targets qwen-smallest; rename to Skywork-8B experiment id.
        if name == "confidence":
            config.name = "confidence_calibration_ece_skywork_gsm8k_mc"

        logger.info("Running %s from %s", name, config_path)
        exp_cls(config).run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    main()
