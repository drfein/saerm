#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.nb.datasets.length import LengthBiasDataset
from src.nb.nullbias.probe import get_rewards_with_nulling
from src.nb.downstream.scoring import RewardModelScorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare baseline vs saved probe-debiased vs LWR, where LWR is fit only on "
            "the original probe-training split (same split_seed/probe_size)."
        )
    )
    parser.add_argument("--raw-data-files", nargs="+", required=True)
    parser.add_argument(
        "--source-file",
        default="",
        help="Optional source override (e.g., data/gsm8k_soln.json) when config.dataset_source path is unavailable.",
    )
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--lwr-frac", type=float, default=0.5)
    parser.add_argument("--lwr-alpha", type=float, default=1.0)
    parser.add_argument("--show-progress", action="store_true")
    return parser.parse_args()


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n)
    return (center - half, center + half)


def paired_exact_mcnemar_one_sided_lower_bad(a_bad: list[int], b_bad: list[int]) -> dict[str, Any]:
    # H1: method a has lower bad-event rate than method b.
    n01 = sum((x == 0 and y == 1) for x, y in zip(a_bad, b_bad))
    n10 = sum((x == 1 and y == 0) for x, y in zip(a_bad, b_bad))
    m = n01 + n10
    if m == 0:
        p_one = 1.0
        p_two = 1.0
    else:
        p_one = sum(math.comb(m, i) * (0.5**m) for i in range(n01, m + 1))
        k = min(n01, n10)
        p_two = min(1.0, 2.0 * sum(math.comb(m, i) * (0.5**m) for i in range(0, k + 1)))
    return {
        "n01_a_good_b_bad": n01,
        "n10_a_bad_b_good": n10,
        "n_discordant": m,
        "p_one_sided_a_lower_bad_rate": p_one,
        "p_two_sided": p_two,
    }


def response_for_variant(raw_example: dict[str, Any], variant: str) -> str | None:
    if variant == "correct":
        return raw_example.get("correct_response")
    if variant == "correct_verbose":
        return raw_example.get("correct_verbose_response")
    if variant == "incorrect":
        return raw_example.get("incorrect_response")
    if variant == "incorrect_short":
        return raw_example.get("incorrect_short_response")
    if variant == "incorrect_long":
        return raw_example.get("incorrect_long_response")
    if variant == "incorrect_correct":
        incorrect = raw_example.get("incorrect_response")
        correct = raw_example.get("correct_response")
        if incorrect is None or correct is None:
            return None
        return f"An incorrect answer is {incorrect}\n\nThe correct answer is {correct}"
    return None


def response_length_words(raw_example: dict[str, Any], variant: str) -> int | None:
    response = response_for_variant(raw_example, variant)
    if response is None:
        return None
    return len(str(response).split())


def hash_score(seed: int, idx: int, example: Any) -> float:
    key = f"{seed}|{idx}|{str(example)}".encode("utf-8")
    h = hashlib.sha256(key).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def split_indices(raw_data: list[Any], probe_size: int, split_seed: int, max_test_examples: int | None) -> tuple[list[int], list[int]]:
    n_total = len(raw_data)
    if probe_size == 0:
        test = list(range(n_total))
        if max_test_examples is not None:
            test = test[:max_test_examples]
        return [], test

    min_test_size = min(max(n_total // 5, 1), 50)
    max_probe_size = max(n_total - min_test_size, 1)
    actual_probe_size = min(probe_size, max_probe_size)

    sorted_indices = sorted(range(n_total), key=lambda idx: hash_score(split_seed, idx, raw_data[idx]))
    probe = sorted_indices[:actual_probe_size]
    test = sorted_indices[actual_probe_size:]
    if max_test_examples is not None:
        test = test[:max_test_examples]
    return probe, test


def fit_lwr_on_probe(
    scorer: RewardModelScorer,
    dataset: LengthBiasDataset,
    variant_names: list[str],
    probe_indices: list[int],
    lwr_frac: float,
    lwr_alpha: float,
    max_length: int,
    show_progress: bool,
) -> tuple[np.ndarray, np.ndarray]:
    probe_texts: list[str | tuple[str, str]] = []
    probe_lengths: list[float] = []

    for idx in probe_indices:
        raw_example = dataset._raw_data[idx]  # type: ignore[index]
        eval_example = dataset._make_eval_example(raw_example, scorer.tokenizer)  # pylint: disable=protected-access
        if eval_example is None:
            continue

        for variant in variant_names:
            if variant not in eval_example.texts:
                continue
            rlen = response_length_words(raw_example, variant)
            if rlen is None:
                continue
            probe_texts.append(eval_example.texts[variant])
            probe_lengths.append(float(rlen))

    if not probe_texts:
        raise RuntimeError("No probe texts collected for LWR fitting.")

    rewards = get_rewards_with_nulling(
        scorer.model,
        scorer.tokenizer,
        probe_texts,
        probe=None,
        alpha=1.0,
        batch_size=scorer.batch_size,
        device=scorer.device,
        max_length=max_length,
        show_progress=show_progress,
    ).cpu().numpy()

    x = np.array(probe_lengths, dtype=float)
    y = np.array(rewards, dtype=float)
    smoothed = lowess(y, x, frac=lwr_frac, return_sorted=True)
    sx = smoothed[:, 0]
    sy = smoothed[:, 1]
    # LWR correction curve f(length); adjusted = baseline - alpha * f(length)
    return sx, sy * lwr_alpha


def summarize_events(events: list[int]) -> dict[str, Any]:
    n = len(events)
    k = int(sum(events))
    rate = (k / n) if n else 0.0
    lo, hi = wilson_ci(k, n)
    return {"n": n, "k_bad": k, "bad_rate": rate, "bad_rate_ci95": [lo, hi]}


def analyze_file(args: argparse.Namespace, raw_file: Path) -> dict[str, Any]:
    raw_data = json.loads(raw_file.read_text())
    cfg = raw_data.get("config", {})

    configured_source = str(cfg.get("dataset_source", ""))
    source_path = Path(configured_source)
    if args.source_file:
        source_path = Path(args.source_file)
    elif not source_path.exists():
        raise FileNotFoundError(
            f"Dataset source does not exist: {configured_source}. "
            "Pass --source-file with a local equivalent."
        )

    dataset = LengthBiasDataset(
        source=str(source_path),
        probe_size=int(cfg.get("probe_size", 500)),
        split_seed=int(cfg.get("split_seed", 42)),
        max_test_examples=cfg.get("max_test_examples"),
    )
    dataset._ensure_loaded()  # pylint: disable=protected-access
    probe_indices_split, test_indices_split = split_indices(
        dataset._raw_data,  # type: ignore[arg-type]
        probe_size=int(cfg.get("probe_size", 500)),
        split_seed=int(cfg.get("split_seed", 42)),
        max_test_examples=cfg.get("max_test_examples"),
    )

    # Map split by question_idx for alignment checks and eval lookup.
    probe_qidx_split = {int(dataset._raw_data[i]["question_idx"]) for i in probe_indices_split}  # type: ignore[index]
    test_qidx_split = {int(dataset._raw_data[i]["question_idx"]) for i in test_indices_split}  # type: ignore[index]
    eval_qidx_from_raw = {int(ex["metadata"]["question_idx"]) for ex in raw_data["examples"]}
    overlap_eval_test = len(eval_qidx_from_raw & test_qidx_split)

    # Prefer exact historical split implied by saved eval rows:
    # probe_qidx = all valid qidx in dataset minus saved eval qidx.
    all_qidx = {int(row["question_idx"]) for row in dataset._raw_data}  # type: ignore[union-attr]
    probe_qidx = all_qidx - eval_qidx_from_raw
    test_qidx = set(eval_qidx_from_raw)

    if probe_qidx & test_qidx:
        raise RuntimeError("Probe/eval contamination detected after saved-run alignment (non-empty intersection).")

    # Convert aligned qidx sets back to indices for probe scoring.
    qidx_to_idx = {int(row["question_idx"]): i for i, row in enumerate(dataset._raw_data)}  # type: ignore[union-attr]
    probe_indices = [qidx_to_idx[qidx] for qidx in probe_qidx]
    test_indices = [qidx_to_idx[qidx] for qidx in test_qidx]

    scorer = RewardModelScorer(
        model_path=str(cfg["model_path"]),
        probe_path=None,
        alpha=1.0,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        trust_remote_code=bool(cfg.get("trust_remote_code", True)),
        torch_dtype=args.torch_dtype,
        force_pair_format=False,
        show_progress=args.show_progress,
    )

    sx, sy = fit_lwr_on_probe(
        scorer=scorer,
        dataset=dataset,
        variant_names=list(raw_data["variant_names"]),
        probe_indices=probe_indices,
        lwr_frac=float(args.lwr_frac),
        lwr_alpha=float(args.lwr_alpha),
        max_length=int(args.max_length),
        show_progress=bool(args.show_progress),
    )

    by_qidx: dict[int, dict[str, Any]] = {
        int(row["question_idx"]): row for row in dataset._raw_data  # type: ignore[union-attr]
    }

    events: dict[str, dict[str, list[int]]] = {
        "correct_long_gt_incorrect": {"baseline": [], "saved_debiased": [], "lwr_2409": []},
        "correct_short_gt_incorrect": {"baseline": [], "saved_debiased": [], "lwr_2409": []},
    }

    for ex in raw_data["examples"]:
        qidx = int(ex["metadata"]["question_idx"])
        raw_ex = by_qidx[qidx]
        rb = ex["baseline_rewards"]
        rn = ex["nulled_rewards"]

        needed = ["correct_verbose", "correct", "incorrect"]
        lengths: dict[str, float] = {}
        for variant in needed:
            rlen = response_length_words(raw_ex, variant)
            if rlen is None:
                break
            lengths[variant] = float(rlen)
        if len(lengths) != len(needed):
            continue

        lwr_scores = {
            variant: float(rb[variant]) - float(np.interp(lengths[variant], sx, sy, left=sy[0], right=sy[-1]))
            for variant in needed
        }

        events["correct_long_gt_incorrect"]["baseline"].append(int(float(rb["correct_verbose"]) > float(rb["incorrect"])))
        events["correct_long_gt_incorrect"]["saved_debiased"].append(
            int(float(rn["correct_verbose"]) > float(rn["incorrect"]))
        )
        events["correct_long_gt_incorrect"]["lwr_2409"].append(int(lwr_scores["correct_verbose"] > lwr_scores["incorrect"]))

        events["correct_short_gt_incorrect"]["baseline"].append(int(float(rb["correct"]) > float(rb["incorrect"])))
        events["correct_short_gt_incorrect"]["saved_debiased"].append(
            int(float(rn["correct"]) > float(rn["incorrect"]))
        )
        events["correct_short_gt_incorrect"]["lwr_2409"].append(int(lwr_scores["correct"] > lwr_scores["incorrect"]))

    metrics: dict[str, Any] = {}
    for metric_name, m in events.items():
        b = m["baseline"]
        n = m["saved_debiased"]
        l = m["lwr_2409"]
        metrics[metric_name] = {
            "baseline": summarize_events(b),
            "saved_debiased": summarize_events(n),
            "lwr_2409": summarize_events(l),
            "paired_tests_one_sided_lower_bad": {
                "saved_debiased_vs_lwr_2409": paired_exact_mcnemar_one_sided_lower_bad(n, l),
                "saved_debiased_vs_baseline": paired_exact_mcnemar_one_sided_lower_bad(n, b),
                "lwr_2409_vs_baseline": paired_exact_mcnemar_one_sided_lower_bad(l, b),
            },
        }

    return {
        "raw_data_file": str(raw_file.resolve()),
        "config_name": cfg.get("name"),
        "reward_model": cfg.get("model_path"),
        "dataset_source_resolved": str(source_path.resolve()),
        "split": {
            "probe_size_requested": int(cfg.get("probe_size", 500)),
            "split_seed": int(cfg.get("split_seed", 42)),
            "n_probe_indices_aligned": len(probe_indices),
            "n_test_indices_aligned": len(test_indices),
            "n_eval_examples_in_raw": len(raw_data["examples"]),
            "reconstructed_split_eval_overlap": overlap_eval_test,
            "reconstructed_split_n_probe": len(probe_indices_split),
            "reconstructed_split_n_test": len(test_indices_split),
        },
        "lwr": {"frac": float(args.lwr_frac), "alpha": float(args.lwr_alpha), "fit_on": "probe_split_only"},
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    analyses = [analyze_file(args, Path(p)) for p in args.raw_data_files]
    output = {
        "method": "lwr_fit_on_probe_split",
        "results": analyses,
    }
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
