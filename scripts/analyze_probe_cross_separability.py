#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze within-family and cross-family linear separability for saved probe activations."
    )
    parser.add_argument("--probe-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_probe_dir(path: Path) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
    probe = torch.load(path / "probe.pt", map_location="cpu").float()
    metadata = json.loads((path / "metadata.json").read_text())
    activations = torch.load(path / "activations.pt", map_location="cpu")
    return probe, metadata, activations


def binary_task(name: str, positive: torch.Tensor, negative: torch.Tensor) -> dict[str, Any]:
    return {
        "name": name,
        "positive": positive.float(),
        "negative": negative.float(),
    }


def build_tasks(probe_root: Path) -> tuple[list[dict[str, Any]], dict[str, list[torch.Tensor]]]:
    sources: dict[str, list[torch.Tensor]] = {}
    tasks: list[dict[str, Any]] = []

    length_dir = probe_root / "length" / "length_skywork_qwen-smallest_gsm8k"
    length_probe, _, length_act = load_probe_dir(length_dir)
    sources["length"] = [length_probe.view(-1)]
    tasks.append(
        binary_task(
            "length",
            length_act["positive_embeddings"],
            length_act["negative_embeddings"],
        )
    )

    syco_dir = probe_root / "sycophancy" / "sycophancy_skywork_qwen-smallest_gsm8k_mc"
    syco_probe, _, syco_act = load_probe_dir(syco_dir)
    sources["sycophancy"] = [syco_probe.view(-1)]
    tasks.append(
        binary_task(
            "sycophancy",
            syco_act["positive_embeddings"],
            syco_act["negative_embeddings"],
        )
    )

    pos_dir = probe_root / "position" / "position_skywork_qwen-smallest_gsm8k"
    pos_probe, _, pos_act = load_probe_dir(pos_dir)
    pos_probe = pos_probe.float()
    sources["position"] = [row.clone() for row in pos_probe]
    embeds_by_pos = pos_act["embeddings_by_position"]
    labels = list(pos_act["position_labels"])
    all_positions = {label: embeds_by_pos[label].float() for label in labels}
    for label in labels:
        negative = torch.cat([all_positions[other] for other in labels if other != label], dim=0)
        tasks.append(binary_task(f"position_{label}_vs_rest", all_positions[label], negative))

    conf_dir = probe_root / "confidence_calibration" / "confidence_calibration_ece_skywork_qwen-smallest_gsm8k_mc"
    conf_probe, conf_meta, conf_act = load_probe_dir(conf_dir)
    conf_probe = conf_probe.float()
    sources["confidence"] = [row.clone() for row in conf_probe]
    per_direction = conf_act["per_direction"]
    for item in per_direction:
        pos = str(item["positive_conf"])
        neg = str(item["negative_conf"])
        tasks.append(
            binary_task(
                f"confidence_{pos}_vs_{neg}",
                item["positive_embeddings"],
                item["negative_embeddings"],
            )
        )

    return tasks, sources


def rank_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores = scores.float().flatten()
    labels = labels.long().flatten()
    pos = int(labels.sum().item())
    neg = int((labels.numel() - pos))
    if pos == 0 or neg == 0:
        return float("nan")
    order = torch.argsort(scores, stable=True)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float32)
    pos_ranks = ranks[labels == 1].sum().item()
    auc = (pos_ranks - pos * (pos + 1) / 2.0) / (pos * neg)
    auc = float(auc)
    return max(auc, 1.0 - auc)


def task_metrics(direction: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> dict[str, float]:
    direction = direction.float().flatten()
    direction = direction / direction.norm().clamp_min(1e-12)
    pos_scores = positive @ direction
    neg_scores = negative @ direction
    scores = torch.cat([pos_scores, neg_scores], dim=0)
    labels = torch.cat(
        [
            torch.ones(pos_scores.shape[0], dtype=torch.long),
            torch.zeros(neg_scores.shape[0], dtype=torch.long),
        ],
        dim=0,
    )
    auc = rank_auc(scores, labels)

    mu_pos = float(pos_scores.mean().item())
    mu_neg = float(neg_scores.mean().item())
    thr = 0.5 * (mu_pos + mu_neg)
    acc_pos = ((pos_scores >= thr).float().mean().item() + (neg_scores < thr).float().mean().item()) / 2.0
    acc_neg = ((pos_scores < thr).float().mean().item() + (neg_scores >= thr).float().mean().item()) / 2.0
    acc = max(acc_pos, acc_neg)

    pos_std = float(pos_scores.std(unbiased=False).item())
    neg_std = float(neg_scores.std(unbiased=False).item())
    pooled = ((pos_std**2 + neg_std**2) / 2.0) ** 0.5
    effect = abs(mu_pos - mu_neg) / max(pooled, 1e-12)

    return {
        "auc": float(auc),
        "balanced_accuracy": float(acc),
        "effect_size_d": float(effect),
        "positive_mean": mu_pos,
        "negative_mean": mu_neg,
    }


def main() -> None:
    args = parse_args()
    probe_root = Path(args.probe_root)
    tasks, sources = build_tasks(probe_root)

    results: dict[str, Any] = {
        "probe_root": str(probe_root),
        "tasks": [task["name"] for task in tasks],
        "sources": {name: len(vectors) for name, vectors in sources.items()},
        "by_source_family": {},
        "best_auc_family_matrix": {},
        "best_balanced_accuracy_family_matrix": {},
    }

    family_task_scores_auc: dict[str, dict[str, list[float]]] = {}
    family_task_scores_acc: dict[str, dict[str, list[float]]] = {}

    for source_family, vectors in sources.items():
        source_out: dict[str, Any] = {"directions": []}
        family_task_scores_auc[source_family] = {}
        family_task_scores_acc[source_family] = {}
        for idx, vec in enumerate(vectors):
            dir_name = f"{source_family}[{idx}]"
            task_out = {}
            for task in tasks:
                metrics = task_metrics(vec, task["positive"], task["negative"])
                task_out[task["name"]] = metrics
                target_family = task["name"].split("_vs_")[0]
                target_family = target_family.split("_")[0]
                family_task_scores_auc[source_family].setdefault(target_family, []).append(metrics["auc"])
                family_task_scores_acc[source_family].setdefault(target_family, []).append(metrics["balanced_accuracy"])
            source_out["directions"].append({"name": dir_name, "tasks": task_out})
        results["by_source_family"][source_family] = source_out

    target_families = ["length", "sycophancy", "confidence", "position"]
    for source_family in sources:
        results["best_auc_family_matrix"][source_family] = {}
        results["best_balanced_accuracy_family_matrix"][source_family] = {}
        for target_family in target_families:
            auc_vals = family_task_scores_auc[source_family].get(target_family, [])
            acc_vals = family_task_scores_acc[source_family].get(target_family, [])
            results["best_auc_family_matrix"][source_family][target_family] = max(auc_vals) if auc_vals else None
            results["best_balanced_accuracy_family_matrix"][source_family][target_family] = max(acc_vals) if acc_vals else None

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
