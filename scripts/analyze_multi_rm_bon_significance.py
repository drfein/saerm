#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SELECTOR_ORDER = [
    "deberta_baseline",
    "deberta_length_nulled",
    "allen_baseline",
    "allen_length_nulled",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute paired significance stats from multi-RM BoN scored outputs."
    )
    parser.add_argument("--details-file", required=True)
    parser.add_argument("--summary-file", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def exact_two_sided_sign_p(num_pos: int, num_neg: int) -> float:
    n = num_pos + num_neg
    if n == 0:
        return 1.0
    k = min(num_pos, num_neg)
    cdf = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    return min(1.0, 2.0 * cdf)


def exact_two_sided_mcnemar_p(b_only: int, a_only: int) -> float:
    return exact_two_sided_sign_p(b_only, a_only)


def selector_words_and_correct(rows: list[dict[str, Any]], selector: str, n: int) -> tuple[list[int], list[bool | None]]:
    words: list[int] = []
    correct: list[bool | None] = []
    for row in rows:
        idx = int(row["best_indices_by_n"][selector][str(n)] if isinstance(row["best_indices_by_n"][selector], dict) and str(n) in row["best_indices_by_n"][selector] else row["best_indices_by_n"][selector][n])
        cand = row["candidates"][idx]
        words.append(int(cand["response_words"]))
        correct.append(cand.get("is_correct"))
    return words, correct


def compare_lengths(a: list[int], b: list[int]) -> dict[str, Any]:
    deltas = [bb - aa for aa, bb in zip(a, b)]
    num_pos = sum(1 for d in deltas if d > 0)
    num_neg = sum(1 for d in deltas if d < 0)
    num_ties = sum(1 for d in deltas if d == 0)
    sorted_d = sorted(deltas)
    n = len(sorted_d)
    return {
        "mean_delta_b_minus_a": sum(deltas) / max(len(deltas), 1),
        "median_delta_b_minus_a": sorted_d[n // 2] if n else 0,
        "num_a_longer": num_neg,
        "num_b_longer": num_pos,
        "num_ties": num_ties,
        "sign_test_p_two_sided": exact_two_sided_sign_p(num_pos, num_neg),
    }


def compare_accuracy(a: list[bool | None], b: list[bool | None]) -> dict[str, Any] | None:
    if any(x is None for x in a) or any(x is None for x in b):
        return None
    a = [bool(x) for x in a]
    b = [bool(x) for x in b]
    a_only = sum(1 for aa, bb in zip(a, b) if aa and not bb)
    b_only = sum(1 for aa, bb in zip(a, b) if bb and not aa)
    both = sum(1 for aa, bb in zip(a, b) if aa and bb)
    neither = sum(1 for aa, bb in zip(a, b) if (not aa) and (not bb))
    return {
        "accuracy_a": sum(a) / len(a),
        "accuracy_b": sum(b) / len(b),
        "delta_accuracy_b_minus_a": (sum(b) - sum(a)) / len(a),
        "a_only_correct": a_only,
        "b_only_correct": b_only,
        "both_correct": both,
        "neither_correct": neither,
        "mcnemar_p_two_sided": exact_two_sided_mcnemar_p(b_only, a_only),
    }


def main() -> None:
    args = parse_args()
    details_rows = load_jsonl(Path(args.details_file))
    summary = json.loads(Path(args.summary_file).read_text())
    n_values = summary["n_values"]

    out: dict[str, Any] = {
        "details_file": str(Path(args.details_file).resolve()),
        "summary_file": str(Path(args.summary_file).resolve()),
        "dataset_name": summary.get("dataset_name"),
        "num_prompts": len(details_rows),
        "n_values": n_values,
        "comparisons": {},
    }

    pairs = [
        ("deberta_baseline", "deberta_length_nulled"),
        ("allen_baseline", "allen_length_nulled"),
        ("deberta_baseline", "allen_baseline"),
        ("deberta_length_nulled", "allen_length_nulled"),
    ]

    for n in n_values:
        out["comparisons"][str(n)] = {}
        cached = {sel: selector_words_and_correct(details_rows, sel, n) for sel in SELECTOR_ORDER}
        for a_name, b_name in pairs:
            a_words, a_correct = cached[a_name]
            b_words, b_correct = cached[b_name]
            comp = {
                "length": compare_lengths(a_words, b_words),
                "accuracy": compare_accuracy(a_correct, b_correct),
            }
            out["comparisons"][str(n)][f"{a_name}__vs__{b_name}"] = comp

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
