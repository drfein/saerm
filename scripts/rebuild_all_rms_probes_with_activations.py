#!/usr/bin/env python3
"""
Rebuild probes for all 5 reward models, saving activations for INLP analysis.

RMs covered:
  allen           allenai/Llama-3.1-8B-Instruct-RM-RB2
  skywork         Skywork/Skywork-Reward-V2-Llama-3.1-8B
  skywork_qwen3   Skywork/Skywork-Reward-V2-Qwen3-8B
  skywork_qwsm    Skywork/Skywork-Reward-V2-Qwen3-0.6B
  deberta         OpenAssistant/reward-model-deberta-v3-large-v2

Bias families: length, position, sycophancy, uncertainty
"""
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

# (family, config_path, experiment_class, rm_key)
RUN_SPECS = [
    # Allen
    ("length",      "experiments/configs/length_allen_gsm8k.yaml",              LengthBiasExperiment,      "allen"),
    ("position",    "experiments/configs/position_allen_gsm8k.yaml",             PositionBiasExperiment,    "allen"),
    ("sycophancy",  "experiments/configs/sycophancy_allen_gsm8k_mc.yaml",        SycophancyBiasExperiment,  "allen"),
    ("uncertainty", "experiments/configs/uncertainty_allen_gsm8k_mc.yaml",       UncertaintyBiasExperiment, "allen"),
    # Skywork-8B
    ("length",      "experiments/configs/length_skywork_gsm8k.yaml",             LengthBiasExperiment,      "skywork"),
    ("position",    "experiments/configs/position_skywork_gsm8k.yaml",           PositionBiasExperiment,    "skywork"),
    ("sycophancy",  "experiments/configs/sycophancy_skywork_gsm8k_mc.yaml",      SycophancyBiasExperiment,  "skywork"),
    ("uncertainty", "experiments/configs/uncertainty_skywork_gsm8k_mc.yaml",     UncertaintyBiasExperiment, "skywork"),
    # Skywork-Qwen3-8B
    ("length",      "experiments/configs/length_skywork_qwen3_gsm8k.yaml",       LengthBiasExperiment,      "skywork_qwen3"),
    ("position",    "experiments/configs/position_skywork_qwen3_gsm8k.yaml",     PositionBiasExperiment,    "skywork_qwen3"),
    ("sycophancy",  "experiments/configs/sycophancy_skywork_qwen3_gsm8k_mc.yaml",SycophancyBiasExperiment,  "skywork_qwen3"),
    ("uncertainty", "experiments/configs/uncertainty_skywork_qwen3_gsm8k_mc.yaml",UncertaintyBiasExperiment,"skywork_qwen3"),
    # Skywork-Qwen3-0.6B
    ("length",      "experiments/configs/length_skywork_qwen-smallest_gsm8k.yaml",       LengthBiasExperiment,      "skywork_qwsm"),
    ("position",    "experiments/configs/position_skywork_qwen-smallest_gsm8k.yaml",     PositionBiasExperiment,    "skywork_qwsm"),
    ("sycophancy",  "experiments/configs/sycophancy_skywork_qwen-smallest_gsm8k_mc.yaml",SycophancyBiasExperiment,  "skywork_qwsm"),
    ("uncertainty", "experiments/configs/uncertainty_skywork_qwen-smallest_gsm8k_mc.yaml",UncertaintyBiasExperiment,"skywork_qwsm"),
    # DeBERTa
    ("length",      "experiments/configs/length_deberta_gsm8k.yaml",              LengthBiasExperiment,      "deberta"),
    ("position",    "experiments/configs/position_deberta_gsm8k.yaml",            PositionBiasExperiment,    "deberta"),
    ("sycophancy",  "experiments/configs/sycophancy_deberta_gsm8k_mc.yaml",       SycophancyBiasExperiment,  "deberta"),
    ("uncertainty", "experiments/configs/uncertainty_deberta_gsm8k_mc.yaml",      UncertaintyBiasExperiment, "deberta"),
]

RM_KEYS = ["allen", "skywork", "skywork_qwen3", "skywork_qwsm", "deberta"]
FAMILIES = ["length", "position", "sycophancy", "uncertainty"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-rm", choices=RM_KEYS, action="append",
                        help="Run only selected RMs (default: all).")
    parser.add_argument("--only-family", choices=FAMILIES, action="append",
                        help="Run only selected bias families (default: all).")
    parser.add_argument("--artifacts-dir", default="artifacts_inlp_rebuild")
    parser.add_argument("--plots-dir", default="plots_inlp_rebuild")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_rms = set(args.only_rm or RM_KEYS)
    selected_families = set(args.only_family or FAMILIES)

    for family, config_rel, exp_cls, rm in RUN_SPECS:
        if rm not in selected_rms or family not in selected_families:
            continue

        config_path = PROJECT_ROOT / config_rel
        if not config_path.exists():
            logger.warning("Config not found, skipping: %s", config_path)
            continue

        config = ExperimentConfig.from_yaml(config_path)
        config.artifacts_dir = args.artifacts_dir
        config.plots_dir = args.plots_dir
        config.device = args.device
        config.extra = dict(config.extra)
        config.extra["save_probe_activations"] = True

        logger.info("Running %s / %s", rm, family)
        exp_cls(config).run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    main()
