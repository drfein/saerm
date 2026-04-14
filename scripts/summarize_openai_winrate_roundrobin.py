#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize round-robin OpenAI win-rate evals.")
    parser.add_argument("--baseline-vs-length-penalty", required=True)
    parser.add_argument("--baseline-vs-length-nulled", required=True)
    parser.add_argument("--length-penalty-vs-length-nulled", required=True)
    parser.add_argument("--output-file", required=True)
    return parser.parse_args()


def load_json(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_pair(summary: dict) -> dict:
    return {
        "judge_model": summary["judge_model"],
        "num_examples": summary["num_examples"],
        "model_a_name": summary["model_a_name"],
        "model_b_name": summary["model_b_name"],
        "model_a_wins": summary["model_a_wins"],
        "model_b_wins": summary["model_b_wins"],
        "ties": summary["ties"],
        "model_a_win_rate": summary["model_a_win_rate"],
        "model_b_win_rate": summary["model_b_win_rate"],
        "tie_rate": summary["tie_rate"],
    }


def main() -> None:
    args = parse_args()
    baseline_vs_penalty = load_json(args.baseline_vs_length_penalty)
    baseline_vs_nulled = load_json(args.baseline_vs_length_nulled)
    penalty_vs_nulled = load_json(args.length_penalty_vs_length_nulled)

    summary = {
        "baseline_vs_length_penalty": extract_pair(baseline_vs_penalty),
        "baseline_vs_length_nulled": extract_pair(baseline_vs_nulled),
        "length_penalty_vs_length_nulled": extract_pair(penalty_vs_nulled),
    }

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
