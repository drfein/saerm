#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.nb.downstream.gsm8k import extract_final_answer, is_correct
from src.nb.downstream.scoring import RewardModelScorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a saved BoN candidate pool with baseline, probe-nulled, and length-penalized selectors.")
    parser.add_argument("--candidate-file", required=True)
    parser.add_argument("--reward-model", required=True)
    parser.add_argument("--probe-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--length-penalty-max-len", type=float, default=256.0)
    parser.add_argument("--length-penalty-scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--force-pair-format", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def quantile(values: list[int], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * p
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(ordered[lo])
    frac = idx - lo
    return float(ordered[lo] * (1 - frac) + ordered[hi] * frac)


def length_penalize_scores(
    baseline_scores: list[float],
    response_words: list[int],
    *,
    max_len: float,
    scale: float,
) -> tuple[list[float], float]:
    sigma = statistics.pstdev(baseline_scores) if len(baseline_scores) > 1 else 0.0
    sigma *= scale
    adjusted = [score + ((1.0 - (words / max_len)) * sigma) for score, words in zip(baseline_scores, response_words)]
    return adjusted, sigma


def main() -> None:
    args = parse_args()
    rows = load_jsonl(Path(args.candidate_file))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    selector_names = ("baseline", "length_nulled", "length_penalty")
    selected_baseline_scores = {name: [] for name in selector_names}
    selected_nulled_scores = {name: [] for name in selector_names}
    selected_lengths = {name: [] for name in selector_names}
    selected_correct = {name: 0 for name in selector_names}
    selection_overlap = {
        "baseline_vs_length_nulled": 0,
        "baseline_vs_length_penalty": 0,
        "length_nulled_vs_length_penalty": 0,
    }
    oracle_correct = 0

    flat_prompts: list[str] = []
    flat_responses: list[str] = []
    counts: list[int] = []
    for row in rows:
        candidates = row["candidates"]
        counts.append(len(candidates))
        flat_prompts.extend([str(row["prompt"])] * len(candidates))
        flat_responses.extend([str(candidate["response"]) for candidate in candidates])

    baseline_all, nulled_all = scorer.score_pairs_both(flat_prompts, flat_responses)

    records_out: list[dict[str, Any]] = []
    offset = 0
    for row, count in zip(rows, counts):
        prompt = str(row["prompt"])
        gold_answer = row.get("gold_answer")
        candidates = []
        baseline_scores: list[float] = []
        nulled_scores: list[float] = []
        response_words: list[int] = []
        for idx in range(count):
            candidate = dict(row["candidates"][idx])
            baseline_score = float(baseline_all[offset + idx])
            nulled_score = float(nulled_all[offset + idx])
            words = int(candidate.get("response_words") or len(str(candidate["response"]).split()))
            candidate["baseline_score"] = baseline_score
            candidate["nulled_score"] = nulled_score
            candidate["response_words"] = words
            candidate["final_answer"] = extract_final_answer(str(candidate["response"]))
            candidates.append(candidate)
            baseline_scores.append(baseline_score)
            nulled_scores.append(nulled_score)
            response_words.append(words)
        offset += count

        length_penalty_scores, sigma = length_penalize_scores(
            baseline_scores,
            response_words,
            max_len=args.length_penalty_max_len,
            scale=args.length_penalty_scale,
        )
        for candidate, penalty_score in zip(candidates, length_penalty_scores):
            candidate["length_penalty_score"] = penalty_score

        best_index = {
            "baseline": max(range(len(candidates)), key=lambda i: candidates[i]["baseline_score"]),
            "length_nulled": max(range(len(candidates)), key=lambda i: candidates[i]["nulled_score"]),
            "length_penalty": max(range(len(candidates)), key=lambda i: candidates[i]["length_penalty_score"]),
        }
        selection_overlap["baseline_vs_length_nulled"] += int(best_index["baseline"] == best_index["length_nulled"])
        selection_overlap["baseline_vs_length_penalty"] += int(best_index["baseline"] == best_index["length_penalty"])
        selection_overlap["length_nulled_vs_length_penalty"] += int(best_index["length_nulled"] == best_index["length_penalty"])

        candidate_correct = [is_correct(str(c["response"]), gold_answer) for c in candidates] if gold_answer is not None else [False] * len(candidates)
        if gold_answer is not None:
            oracle_correct += int(any(candidate_correct))

        for selector in selector_names:
            chosen_idx = best_index[selector]
            selected_baseline_scores[selector].append(float(candidates[chosen_idx]["baseline_score"]))
            selected_nulled_scores[selector].append(float(candidates[chosen_idx]["nulled_score"]))
            selected_lengths[selector].append(int(candidates[chosen_idx]["response_words"]))
            if gold_answer is not None:
                selected_correct[selector] += int(candidate_correct[chosen_idx])

        for candidate, ok in zip(candidates, candidate_correct):
            candidate["is_correct"] = ok if gold_answer is not None else None

        records_out.append(
            {
                "index": row.get("index"),
                "prompt": prompt,
                "gold_answer": gold_answer,
                "selector_best_index": best_index,
                "selector_best_response": {name: candidates[best_index[name]]["response"] for name in selector_names},
                "length_penalty_sigma": sigma,
                "candidates": candidates,
            }
        )

    num_prompts = len(records_out)
    summary: dict[str, Any] = {
        "num_prompts": num_prompts,
        "n": counts[0] if counts else 0,
        "reward_model": args.reward_model,
        "probe_file": str(Path(args.probe_file).resolve()),
        "alpha": args.alpha,
        "length_penalty_max_len": args.length_penalty_max_len,
        "length_penalty_scale": args.length_penalty_scale,
        "selection_overlap_rate": {name: count / max(num_prompts, 1) for name, count in selection_overlap.items()},
    }
    for selector in selector_names:
        summary[f"{selector}_selected_baseline_score_mean"] = mean(selected_baseline_scores[selector])
        summary[f"{selector}_selected_nulled_score_mean"] = mean(selected_nulled_scores[selector])
        summary[f"{selector}_selected_response_words_mean"] = mean([float(x) for x in selected_lengths[selector]])
        summary[f"{selector}_selected_response_words_stdev"] = statistics.pstdev(selected_lengths[selector]) if selected_lengths[selector] else 0.0
        summary[f"{selector}_selected_response_words_variance"] = statistics.pvariance(selected_lengths[selector]) if selected_lengths[selector] else 0.0
        summary[f"{selector}_selected_response_words_min"] = min(selected_lengths[selector]) if selected_lengths[selector] else 0
        summary[f"{selector}_selected_response_words_p25"] = quantile(selected_lengths[selector], 0.25)
        summary[f"{selector}_selected_response_words_median"] = quantile(selected_lengths[selector], 0.5)
        summary[f"{selector}_selected_response_words_p75"] = quantile(selected_lengths[selector], 0.75)
        summary[f"{selector}_selected_response_words_max"] = max(selected_lengths[selector]) if selected_lengths[selector] else 0

    if rows and rows[0].get("gold_answer") is not None:
        for selector in selector_names:
            summary[f"{selector}_accuracy"] = selected_correct[selector] / max(num_prompts, 1)
        summary["oracle_accuracy"] = oracle_correct / max(num_prompts, 1)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (output_dir / "details.jsonl").open("w", encoding="utf-8") as f:
        for record in records_out:
            f.write(json.dumps(record) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
