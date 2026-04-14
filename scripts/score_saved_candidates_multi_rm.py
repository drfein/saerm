#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from src.nb.downstream.gsm8k import extract_final_answer, is_correct
from src.nb.downstream.scoring import RewardModelScorer


SELECTOR_ORDER = [
    "deberta_baseline",
    "deberta_length_nulled",
    "allen_baseline",
    "allen_length_nulled",
]

SELECTOR_STYLES = {
    "deberta_baseline": ("DeBERTa base", "#1f77b4"),
    "deberta_length_nulled": ("DeBERTa debiased", "#ff7f0e"),
    "allen_baseline": ("Allen base", "#2ca02c"),
    "allen_length_nulled": ("Allen debiased", "#d62728"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a saved candidate pool with DeBERTa and Allen baseline/debiased selectors, then plot mean selected length vs N."
    )
    parser.add_argument("--candidate-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--deberta-model", default="OpenAssistant/reward-model-deberta-v3-large-v2")
    parser.add_argument("--deberta-probe-file", required=True)
    parser.add_argument("--allen-model", default="allenai/Llama-3.1-8B-Instruct-RM-RB2")
    parser.add_argument("--allen-probe-file", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--n-values", default="1,2,4,8,16,32,64")
    parser.add_argument("--force-pair-format", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def build_flat_batches(rows: list[dict[str, Any]]) -> tuple[list[str], list[str], list[int]]:
    flat_prompts: list[str] = []
    flat_responses: list[str] = []
    counts: list[int] = []
    for row in rows:
        prompt = str(row["prompt"])
        candidates = row["candidates"]
        counts.append(len(candidates))
        flat_prompts.extend([prompt] * len(candidates))
        flat_responses.extend([str(candidate["response"]) for candidate in candidates])
    return flat_prompts, flat_responses, counts


def score_family(
    *,
    family_name: str,
    model_path: str,
    probe_path: str,
    prompts: list[str],
    responses: list[str],
    device: str,
    batch_size: int,
    max_length: int,
    torch_dtype: str,
    alpha: float,
    force_pair_format: bool,
) -> tuple[list[float], list[float]]:
    print(
        f"[score] {family_name}: scoring {len(responses)} candidates "
        f"(batch_size={batch_size}, max_length={max_length})",
        flush=True,
    )
    scorer = RewardModelScorer(
        model_path=model_path,
        probe_path=probe_path,
        alpha=alpha,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        torch_dtype=torch_dtype,
        force_pair_format=force_pair_format,
        show_progress=True,
    )
    return scorer.score_pairs_both(prompts, responses)


def plot_curves(output_path: Path, dataset_name: str, curves: dict[str, list[dict[str, float]]]) -> None:
    plt.figure(figsize=(8, 5))
    for selector in SELECTOR_ORDER:
        label, color = SELECTOR_STYLES[selector]
        xs = [point["n"] for point in curves[selector]]
        ys = [point["mean_response_words"] for point in curves[selector]]
        plt.plot(xs, ys, marker="o", linewidth=2, markersize=4, label=label, color=color)
    plt.xlabel("Best-of-N")
    plt.ylabel("Mean selected length (words)")
    plt.title(f"{dataset_name}: mean selected length vs N")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(Path(args.candidate_file))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_values = [int(x) for x in args.n_values.split(",") if x.strip()]
    print(
        f"[load] dataset={args.dataset_name} prompts={len(rows)} "
        f"n_values={n_values} candidate_file={args.candidate_file}",
        flush=True,
    )

    flat_prompts, flat_responses, counts = build_flat_batches(rows)
    deberta_base_all, deberta_nulled_all = score_family(
        family_name="DeBERTa",
        model_path=args.deberta_model,
        probe_path=args.deberta_probe_file,
        prompts=flat_prompts,
        responses=flat_responses,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        torch_dtype=args.torch_dtype,
        alpha=args.alpha,
        force_pair_format=args.force_pair_format,
    )
    allen_base_all, allen_nulled_all = score_family(
        family_name="Allen",
        model_path=args.allen_model,
        probe_path=args.allen_probe_file,
        prompts=flat_prompts,
        responses=flat_responses,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        torch_dtype=args.torch_dtype,
        alpha=args.alpha,
        force_pair_format=args.force_pair_format,
    )
    print("[aggregate] computing winners and summary curves", flush=True)

    selector_lengths = {name: {n: [] for n in n_values} for name in SELECTOR_ORDER}
    selector_accuracy = {name: {n: 0 for n in n_values} for name in SELECTOR_ORDER}
    records_out: list[dict[str, Any]] = []

    offset = 0
    for row, count in zip(rows, counts):
        candidates = []
        gold_answer = row.get("gold_answer")
        for candidate_idx in range(count):
            idx = offset + candidate_idx
            candidate = dict(row["candidates"][candidate_idx])
            candidate["response_words"] = int(candidate.get("response_words") or len(str(candidate["response"]).split()))
            candidate["deberta_baseline_score"] = float(deberta_base_all[idx])
            candidate["deberta_length_nulled_score"] = float(deberta_nulled_all[idx])
            candidate["allen_baseline_score"] = float(allen_base_all[idx])
            candidate["allen_length_nulled_score"] = float(allen_nulled_all[idx])
            candidate["final_answer"] = extract_final_answer(str(candidate["response"]))
            if gold_answer is not None:
                candidate["is_correct"] = is_correct(str(candidate["response"]), gold_answer)
            else:
                candidate["is_correct"] = None
            candidates.append(candidate)
        offset += count

        best_indices_by_n: dict[str, dict[int, int]] = {name: {} for name in SELECTOR_ORDER}
        for n in n_values:
            capped = min(n, len(candidates))
            subset = candidates[:capped]
            for selector in SELECTOR_ORDER:
                idx = max(range(capped), key=lambda i: subset[i][f"{selector}_score"])
                best_indices_by_n[selector][n] = idx
                selector_lengths[selector][n].append(int(subset[idx]["response_words"]))
                if gold_answer is not None and subset[idx]["is_correct"]:
                    selector_accuracy[selector][n] += 1

        records_out.append(
            {
                "index": row.get("index"),
                "prompt": row.get("prompt"),
                "gold_answer": gold_answer,
                "candidates": candidates,
                "best_indices_by_n": best_indices_by_n,
            }
        )

    summary: dict[str, Any] = {
        "dataset_name": args.dataset_name,
        "candidate_file": str(Path(args.candidate_file).resolve()),
        "num_prompts": len(rows),
        "n_values": n_values,
        "curves": {},
    }

    for selector in SELECTOR_ORDER:
        curve = []
        for n in n_values:
            point = {
                "n": n,
                "mean_response_words": mean([float(x) for x in selector_lengths[selector][n]]),
            }
            if rows and rows[0].get("gold_answer") is not None:
                point["accuracy"] = selector_accuracy[selector][n] / max(len(rows), 1)
            curve.append(point)
        summary["curves"][selector] = curve

    plot_curves(output_dir / "mean_length_vs_n.png", args.dataset_name, summary["curves"])

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (output_dir / "details.jsonl").open("w", encoding="utf-8") as f:
        for row in records_out:
            f.write(json.dumps(row) + "\n")

    print(f"[done] wrote summary to {output_dir / 'summary.json'}", flush=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
