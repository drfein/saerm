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
        description="Rescore a saved candidate pool with a baseline RM and two probe-nulled variants."
    )
    parser.add_argument("--candidate-file", required=True)
    parser.add_argument("--reward-model", required=True)
    parser.add_argument("--probe-a-file", required=True)
    parser.add_argument("--probe-a-name", required=True)
    parser.add_argument("--probe-b-file", required=True)
    parser.add_argument("--probe-b-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--alpha-a", type=float, default=1.0)
    parser.add_argument("--alpha-b", type=float, default=1.0)
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

    flat_prompts: list[str] = []
    flat_responses: list[str] = []
    counts: list[int] = []
    for row in rows:
        prompt = str(row["prompt"])
        candidates = row["candidates"]
        counts.append(len(candidates))
        flat_prompts.extend([prompt] * len(candidates))
        flat_responses.extend([str(candidate["response"]) for candidate in candidates])

    scorer_a = RewardModelScorer(
        model_path=args.reward_model,
        probe_path=args.probe_a_file,
        alpha=args.alpha_a,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        torch_dtype=args.torch_dtype,
        force_pair_format=args.force_pair_format,
    )
    baseline_all, probe_a_all = scorer_a.score_pairs_both(flat_prompts, flat_responses)
    del scorer_a

    scorer_b = RewardModelScorer(
        model_path=args.reward_model,
        probe_path=args.probe_b_file,
        alpha=args.alpha_b,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        torch_dtype=args.torch_dtype,
        force_pair_format=args.force_pair_format,
    )
    baseline_check_all, probe_b_all = scorer_b.score_pairs_both(flat_prompts, flat_responses)
    del scorer_b

    selector_names = ["baseline", args.probe_a_name, args.probe_b_name]
    selector_lengths: dict[str, list[int]] = {name: [] for name in selector_names}
    selector_scores: dict[str, list[float]] = {name: [] for name in selector_names}
    overlaps = {
        f"baseline_vs_{args.probe_a_name}": 0,
        f"baseline_vs_{args.probe_b_name}": 0,
        f"{args.probe_a_name}_vs_{args.probe_b_name}": 0,
    }

    rescored_rows: list[dict[str, Any]] = []
    offset = 0
    for row, count in zip(rows, counts):
        candidates = []
        for candidate_idx, candidate in enumerate(row["candidates"]):
            idx = offset + candidate_idx
            candidate_record = dict(candidate)
            candidate_record["qwen3_baseline_score"] = float(baseline_all[idx])
            candidate_record[f"{args.probe_a_name}_score"] = float(probe_a_all[idx])
            candidate_record[f"{args.probe_b_name}_score"] = float(probe_b_all[idx])
            candidate_record["qwen3_baseline_score_check"] = float(baseline_check_all[idx])
            if "response_words" not in candidate_record:
                candidate_record["response_words"] = len(str(candidate_record["response"]).split())
            candidates.append(candidate_record)
        offset += count

        best_idx = {
            "baseline": max(range(len(candidates)), key=lambda i: candidates[i]["qwen3_baseline_score"]),
            args.probe_a_name: max(range(len(candidates)), key=lambda i: candidates[i][f"{args.probe_a_name}_score"]),
            args.probe_b_name: max(range(len(candidates)), key=lambda i: candidates[i][f"{args.probe_b_name}_score"]),
        }
        overlaps[f"baseline_vs_{args.probe_a_name}"] += int(best_idx["baseline"] == best_idx[args.probe_a_name])
        overlaps[f"baseline_vs_{args.probe_b_name}"] += int(best_idx["baseline"] == best_idx[args.probe_b_name])
        overlaps[f"{args.probe_a_name}_vs_{args.probe_b_name}"] += int(
            best_idx[args.probe_a_name] == best_idx[args.probe_b_name]
        )

        for selector_name, idx in best_idx.items():
            selector_lengths[selector_name].append(int(candidates[idx]["response_words"]))
            score_key = "qwen3_baseline_score" if selector_name == "baseline" else f"{selector_name}_score"
            selector_scores[selector_name].append(float(candidates[idx][score_key]))

        rescored_rows.append(
            {
                "index": row.get("index"),
                "prompt": row.get("prompt"),
                "selector_best_index": best_idx,
                "selector_best_response": {
                    selector_name: candidates[idx]["response"] for selector_name, idx in best_idx.items()
                },
                "candidates": candidates,
            }
        )

    summary: dict[str, Any] = {
        "candidate_file": str(Path(args.candidate_file).resolve()),
        "reward_model": args.reward_model,
        "probe_a_file": str(Path(args.probe_a_file).resolve()),
        "probe_b_file": str(Path(args.probe_b_file).resolve()),
        "num_prompts": len(rows),
        "n": counts[0] if counts else 0,
        "selection_overlap_rate": {key: value / max(len(rows), 1) for key, value in overlaps.items()},
    }
    for selector_name in selector_names:
        summary[f"{selector_name}_selected_score_mean"] = mean(selector_scores[selector_name])
        summary[f"{selector_name}_selected_response_words"] = summarize_lengths(selector_lengths[selector_name])

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (output_dir / "details.jsonl").open("w", encoding="utf-8") as f:
        for row in rescored_rows:
            f.write(json.dumps(row) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
