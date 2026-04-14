#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from src.nb.downstream.scoring import RewardModelScorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rescore a saved BoN candidate pool with a baseline and probe-nulled reward model."
    )
    parser.add_argument("--candidate-file", required=True)
    parser.add_argument("--reward-model", required=True)
    parser.add_argument("--probe-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--force-pair-format", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def quantile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(ordered[lo])
    frac = idx - lo
    return float(ordered[lo] * (1 - frac) + ordered[hi] * frac)


def summarize_lengths(lengths: list[int]) -> dict[str, float]:
    return {
        "mean": mean([float(x) for x in lengths]),
        "stdev": statistics.pstdev(lengths) if lengths else 0.0,
        "variance": statistics.pvariance(lengths) if lengths else 0.0,
        "min": min(lengths) if lengths else 0.0,
        "p25": quantile(lengths, 0.25),
        "median": quantile(lengths, 0.5),
        "p75": quantile(lengths, 0.75),
        "max": max(lengths) if lengths else 0.0,
    }


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(Path(args.candidate_file))
    scorer = RewardModelScorer(
        model_path=args.reward_model,
        probe_path=args.probe_file,
        alpha=args.alpha,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        torch_dtype=args.torch_dtype,
        force_pair_format=args.force_pair_format,
    )

    flat_prompts: list[str] = []
    flat_responses: list[str] = []
    counts: list[int] = []
    for row in rows:
        prompt = str(row["prompt"])
        candidates = row["candidates"]
        counts.append(len(candidates))
        flat_prompts.extend([prompt] * len(candidates))
        flat_responses.extend([str(candidate["response"]) for candidate in candidates])

    baseline_all, nulled_all = scorer.score_pairs_both(flat_prompts, flat_responses)

    selector_lengths: dict[str, list[int]] = {"baseline": [], "length_nulled": []}
    selector_baseline_scores: dict[str, list[float]] = {"baseline": [], "length_nulled": []}
    selector_nulled_scores: dict[str, list[float]] = {"baseline": [], "length_nulled": []}
    overlap = 0

    offset = 0
    rescored_rows: list[dict[str, Any]] = []
    for row, count in zip(rows, counts):
        candidates = []
        for candidate_idx, candidate in enumerate(row["candidates"]):
            baseline_score = float(baseline_all[offset + candidate_idx])
            nulled_score = float(nulled_all[offset + candidate_idx])
            candidate_record = dict(candidate)
            candidate_record["shp_baseline_score"] = baseline_score
            candidate_record["shp_nulled_score"] = nulled_score
            if "response_words" not in candidate_record:
                candidate_record["response_words"] = len(str(candidate_record["response"]).split())
            candidates.append(candidate_record)
        offset += count

        baseline_best = max(range(len(candidates)), key=lambda i: candidates[i]["shp_baseline_score"])
        nulled_best = max(range(len(candidates)), key=lambda i: candidates[i]["shp_nulled_score"])
        overlap += int(baseline_best == nulled_best)

        for selector, idx in (("baseline", baseline_best), ("length_nulled", nulled_best)):
            selector_lengths[selector].append(int(candidates[idx]["response_words"]))
            selector_baseline_scores[selector].append(float(candidates[idx]["shp_baseline_score"]))
            selector_nulled_scores[selector].append(float(candidates[idx]["shp_nulled_score"]))

        rescored_rows.append(
            {
                "index": row.get("index"),
                "prompt": row.get("prompt"),
                "selector_best_index": {
                    "baseline": baseline_best,
                    "length_nulled": nulled_best,
                },
                "selector_best_response": {
                    "baseline": candidates[baseline_best]["response"],
                    "length_nulled": candidates[nulled_best]["response"],
                },
                "candidates": candidates,
            }
        )

    summary = {
        "candidate_file": str(Path(args.candidate_file).resolve()),
        "reward_model": args.reward_model,
        "probe_file": str(Path(args.probe_file).resolve()),
        "num_prompts": len(rows),
        "n": counts[0] if counts else 0,
        "selection_overlap_rate": overlap / max(len(rows), 1),
        "baseline_selected_baseline_score_mean": mean(selector_baseline_scores["baseline"]),
        "baseline_selected_nulled_score_mean": mean(selector_nulled_scores["baseline"]),
        "length_nulled_selected_baseline_score_mean": mean(selector_baseline_scores["length_nulled"]),
        "length_nulled_selected_nulled_score_mean": mean(selector_nulled_scores["length_nulled"]),
        "baseline_selected_response_words": summarize_lengths(selector_lengths["baseline"]),
        "length_nulled_selected_response_words": summarize_lengths(selector_lengths["length_nulled"]),
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (output_dir / "details.jsonl").open("w", encoding="utf-8") as f:
        for row in rescored_rows:
            f.write(json.dumps(row) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
