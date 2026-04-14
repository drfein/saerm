#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class TaskData:
    name: str
    positive: torch.Tensor
    negative: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Iterative null-space analysis for saved activations. "
            "Learns direction 1 on train split, nulls it, then checks if direction 2 "
            "still separates held-out examples."
        )
    )
    parser.add_argument("--probe-root", required=True, help="Root probes directory containing bias subdirs.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--max-iters", type=int, default=5)
    parser.add_argument(
        "--families",
        nargs="*",
        default=["length", "sycophancy", "position", "uncertainty", "confidence_calibration"],
        help="Probe families to scan under --probe-root.",
    )
    return parser.parse_args()


def _orthonormal_basis(directions: list[torch.Tensor]) -> torch.Tensor:
    if not directions:
        return torch.zeros(0, 1)
    basis: list[torch.Tensor] = []
    for d in directions:
        v = d.float().clone()
        for b in basis:
            v = v - (v @ b) * b
        n = v.norm()
        if n > 1e-8:
            basis.append(v / n)
    if not basis:
        return torch.zeros(0, directions[0].numel())
    return torch.stack(basis, dim=0)


def _null_with_basis(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    if basis.numel() == 0 or basis.shape[0] == 0:
        return x
    return x - (x @ basis.T) @ basis


def _rank_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores = scores.float().flatten()
    labels = labels.long().flatten()
    n_pos = int(labels.sum().item())
    n_neg = int(labels.numel() - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(scores, stable=True)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float32)
    pos_rank_sum = ranks[labels == 1].sum().item()
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    auc = float(auc)
    return max(auc, 1.0 - auc)


def _balanced_acc(scores_pos: torch.Tensor, scores_neg: torch.Tensor) -> float:
    mu_pos = float(scores_pos.mean().item())
    mu_neg = float(scores_neg.mean().item())
    thr = 0.5 * (mu_pos + mu_neg)
    acc_a = 0.5 * (
        float((scores_pos >= thr).float().mean().item()) +
        float((scores_neg < thr).float().mean().item())
    )
    acc_b = 0.5 * (
        float((scores_pos < thr).float().mean().item()) +
        float((scores_neg >= thr).float().mean().item())
    )
    return max(acc_a, acc_b)


def _split_class(x: torch.Tensor, train_frac: float, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    n = x.shape[0]
    perm = torch.randperm(n, generator=g)
    n_train = int(round(train_frac * n))
    n_train = max(1, min(n - 1, n_train))
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    return x[train_idx], x[test_idx]


def _evaluate_iterative(
    task: TaskData,
    *,
    seed: int,
    train_frac: float,
    max_iters: int,
) -> dict[str, Any]:
    g = torch.Generator().manual_seed(seed)
    pos = task.positive.float()
    neg = task.negative.float()

    pos_tr, pos_te = _split_class(pos, train_frac=train_frac, g=g)
    neg_tr, neg_te = _split_class(neg, train_frac=train_frac, g=g)

    directions: list[torch.Tensor] = []
    per_iter: list[dict[str, Any]] = []

    for step in range(1, max_iters + 1):
        basis_prev = _orthonormal_basis(directions)
        tr_pos_res = _null_with_basis(pos_tr, basis_prev)
        tr_neg_res = _null_with_basis(neg_tr, basis_prev)
        te_pos_res = _null_with_basis(pos_te, basis_prev)
        te_neg_res = _null_with_basis(neg_te, basis_prev)

        d = tr_pos_res.mean(dim=0) - tr_neg_res.mean(dim=0)
        d_norm = float(d.norm().item())
        if d_norm < 1e-8:
            per_iter.append(
                {
                    "iter": step,
                    "direction_norm": d_norm,
                    "auc_test": None,
                    "balanced_acc_test": None,
                    "separation_mean_diff_test": None,
                }
            )
            break
        d = d / d.norm().clamp_min(1e-12)
        directions.append(d)

        s_pos = te_pos_res @ d
        s_neg = te_neg_res @ d
        s_all = torch.cat([s_pos, s_neg], dim=0)
        y_all = torch.cat(
            [torch.ones(s_pos.shape[0], dtype=torch.long), torch.zeros(s_neg.shape[0], dtype=torch.long)],
            dim=0,
        )

        per_iter.append(
            {
                "iter": step,
                "direction_norm": d_norm,
                "auc_test": _rank_auc(s_all, y_all),
                "balanced_acc_test": _balanced_acc(s_pos, s_neg),
                "separation_mean_diff_test": float(abs(s_pos.mean().item() - s_neg.mean().item())),
            }
        )

    auc1 = per_iter[0]["auc_test"] if per_iter else None
    auc2 = per_iter[1]["auc_test"] if len(per_iter) > 1 else None
    acc1 = per_iter[0]["balanced_acc_test"] if per_iter else None
    acc2 = per_iter[1]["balanced_acc_test"] if len(per_iter) > 1 else None

    rank1_linear_evidence = False
    if auc1 is not None and auc2 is not None and acc1 is not None and acc2 is not None:
        rank1_linear_evidence = bool((auc1 >= 0.6) and (auc2 <= 0.55) and (acc2 <= 0.55))

    return {
        "task": task.name,
        "n_positive": int(pos.shape[0]),
        "n_negative": int(neg.shape[0]),
        "train_frac": train_frac,
        "max_iters": max_iters,
        "per_iter": per_iter,
        "iter1_auc": auc1,
        "iter2_auc": auc2,
        "iter1_balanced_acc": acc1,
        "iter2_balanced_acc": acc2,
        "delta_auc_1_to_2": None if (auc1 is None or auc2 is None) else float(auc2 - auc1),
        "rank1_linear_evidence": rank1_linear_evidence,
    }


def _load_from_activation_payload(family: str, experiment: str, payload: dict[str, Any]) -> list[TaskData]:
    tasks: list[TaskData] = []

    if "positive_embeddings" in payload and "negative_embeddings" in payload:
        tasks.append(
            TaskData(
                f"{family}:{experiment}",
                payload["positive_embeddings"].float(),
                payload["negative_embeddings"].float(),
            )
        )
        return tasks

    if "embeddings_by_position" in payload and "position_labels" in payload:
        by_pos = {k: v.float() for k, v in payload["embeddings_by_position"].items()}
        labels = list(payload["position_labels"])
        for label in labels:
            positive = by_pos[label]
            negative = torch.cat([by_pos[other] for other in labels if other != label], dim=0)
            tasks.append(TaskData(f"{family}:{experiment}:{label}_vs_rest", positive, negative))
        return tasks

    if "per_direction" in payload:
        for i, item in enumerate(payload["per_direction"]):
            pos_name = str(item.get("positive_conf", "pos"))
            neg_name = str(item.get("negative_conf", "neg"))
            tasks.append(
                TaskData(
                    f"{family}:{experiment}:{pos_name}_vs_{neg_name}_{i}",
                    item["positive_embeddings"].float(),
                    item["negative_embeddings"].float(),
                )
            )
        return tasks

    return tasks


def _load_task_files(probe_root: Path, families: list[str]) -> list[TaskData]:
    tasks: list[TaskData] = []
    for family in families:
        family_dir = probe_root / family
        if not family_dir.exists():
            continue
        for act_path in sorted(family_dir.glob("*/activations.pt")):
            experiment = act_path.parent.name
            payload = torch.load(act_path, map_location="cpu")
            tasks.extend(_load_from_activation_payload(family, experiment, payload))
    return tasks


def main() -> None:
    args = parse_args()
    probe_root = Path(args.probe_root)
    tasks = _load_task_files(probe_root, args.families)
    if not tasks:
        raise ValueError(f"No known activation files found under: {probe_root}")

    results = []
    for task in tasks:
        results.append(
            _evaluate_iterative(
                task,
                seed=args.seed,
                train_frac=args.train_frac,
                max_iters=args.max_iters,
            )
        )

    summary = {
        "probe_root": str(probe_root.resolve()),
        "seed": args.seed,
        "train_frac": args.train_frac,
        "max_iters": args.max_iters,
        "n_tasks": len(results),
        "results": results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
