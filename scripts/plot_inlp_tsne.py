#!/usr/bin/env python3
"""
Run INLP analysis and generate a panel of t-SNE plots showing linear separability
before and after null-space projection, for each bias type × RM.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

PROJECT_ROOT = Path(os.environ.get("SAERM_ROOT", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

FAMILIES = ["length", "sycophancy", "position", "uncertainty", "confidence_calibration"]

FAMILY_LABELS = {
    "length": "Length",
    "sycophancy": "Sycophancy",
    "position": "Position",
    "uncertainty": "Uncertainty",
    "confidence_calibration": "Conf. Calib.",
}


# ── INLP helpers (same logic as analyze_iterative_nulling.py) ────────────────

def _orthonormal_basis(directions: list[torch.Tensor]) -> torch.Tensor:
    basis: list[torch.Tensor] = []
    for d in directions:
        v = d.float().clone()
        for b in basis:
            v = v - (v @ b) * b
        n = v.norm()
        if n > 1e-8:
            basis.append(v / n)
    if not basis:
        return torch.zeros(0, 1)
    return torch.stack(basis, dim=0)


def _null_with_basis(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    if basis.numel() == 0 or basis.shape[0] == 0:
        return x
    return x - (x @ basis.T) @ basis


def _rank_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores = scores.float().flatten()
    labels = labels.long().flatten()
    n_pos = int(labels.sum())
    n_neg = int(labels.numel() - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(scores, stable=True)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float32)
    pos_rank_sum = float(ranks[labels == 1].sum())
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return max(float(auc), 1.0 - float(auc))


def run_inlp(pos: torch.Tensor, neg: torch.Tensor, seed: int = 42, train_frac: float = 0.7, max_iters: int = 5):
    """Returns per_iter list and the first null direction (for projection)."""
    g = torch.Generator().manual_seed(seed)
    pos, neg = pos.float(), neg.float()

    def split(x):
        n = x.shape[0]
        perm = torch.randperm(n, generator=g)
        n_tr = max(1, min(n - 1, int(round(train_frac * n))))
        return x[perm[:n_tr]], x[perm[n_tr:]]

    pos_tr, pos_te = split(pos)
    neg_tr, neg_te = split(neg)

    directions: list[torch.Tensor] = []
    per_iter = []

    for step in range(1, max_iters + 1):
        basis = _orthonormal_basis(directions)
        tr_pos_r = _null_with_basis(pos_tr, basis)
        tr_neg_r = _null_with_basis(neg_tr, basis)
        te_pos_r = _null_with_basis(pos_te, basis)
        te_neg_r = _null_with_basis(neg_te, basis)

        d = tr_pos_r.mean(0) - tr_neg_r.mean(0)
        if float(d.norm()) < 1e-8:
            break
        d = d / d.norm()
        directions.append(d)

        s_pos = te_pos_r @ d
        s_neg = te_neg_r @ d
        s_all = torch.cat([s_pos, s_neg])
        y_all = torch.cat([torch.ones(len(s_pos), dtype=torch.long), torch.zeros(len(s_neg), dtype=torch.long)])
        auc = _rank_auc(s_all, y_all)
        per_iter.append({"iter": step, "auc_test": auc})

    return per_iter, directions


# ── t-SNE helpers ─────────────────────────────────────────────────────────────

def tsne_2d(embeddings: np.ndarray, seed: int = 42) -> np.ndarray:
    n = embeddings.shape[0]
    perplexity = min(30, n // 4)
    return TSNE(n_components=2, perplexity=perplexity, random_state=seed, n_jobs=-1).fit_transform(embeddings)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_tasks(probe_root: Path, families: list[str]) -> list[dict]:
    tasks = []
    for family in families:
        family_dir = probe_root / family
        if not family_dir.exists():
            continue
        for act_path in sorted(family_dir.glob("*/activations.pt")):
            experiment = act_path.parent.name
            payload = torch.load(act_path, map_location="cpu")

            if "positive_embeddings" in payload and "negative_embeddings" in payload:
                tasks.append({
                    "name": f"{family}:{experiment}",
                    "family": family,
                    "experiment": experiment,
                    "positive": payload["positive_embeddings"].float(),
                    "negative": payload["negative_embeddings"].float(),
                })
            elif "embeddings_by_position" in payload and "position_labels" in payload:
                by_pos = {k: v.float() for k, v in payload["embeddings_by_position"].items()}
                labels = list(payload["position_labels"])
                for label in labels:
                    pos = by_pos[label]
                    neg = torch.cat([by_pos[o] for o in labels if o != label], dim=0)
                    tasks.append({
                        "name": f"{family}:{experiment}:{label}_vs_rest",
                        "family": family,
                        "experiment": experiment,
                        "positive": pos,
                        "negative": neg,
                    })
    return tasks


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="INLP table + t-SNE panel.")
    parser.add_argument("--probe-root", required=True)
    parser.add_argument("--output-json", required=True, help="Path for INLP results JSON.")
    parser.add_argument("--output-plot", required=True, help="Path for t-SNE panel PNG.")
    parser.add_argument("--families", nargs="*", default=FAMILIES)
    parser.add_argument("--max-iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--max-points-tsne", type=int, default=2000, help="Max points per class for t-SNE.")
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    probe_root = Path(args.probe_root)
    tasks = load_tasks(probe_root, args.families)
    if not tasks:
        raise ValueError(f"No activation files found under {probe_root}")
    logger.info("Loaded %d tasks", len(tasks))

    inlp_results = []
    tsne_data = []  # (task_name, family, orig_xy, proj_xy, labels, auc1, auc2)

    for task in tasks:
        pos, neg = task["positive"], task["negative"]
        per_iter, directions = run_inlp(pos, neg, seed=args.seed, train_frac=args.train_frac, max_iters=args.max_iters)

        auc1 = per_iter[0]["auc_test"] if per_iter else None
        auc2 = per_iter[1]["auc_test"] if len(per_iter) > 1 else None

        inlp_results.append({
            "task": task["name"],
            "family": task["family"],
            "n_positive": int(pos.shape[0]),
            "n_negative": int(neg.shape[0]),
            "per_iter": per_iter,
            "iter1_auc": auc1,
            "iter2_auc": auc2,
            "delta_auc": None if (auc1 is None or auc2 is None) else float(auc2 - auc1),
            "rank1_linear_evidence": bool(
                auc1 is not None and auc2 is not None and auc1 >= 0.6 and auc2 <= 0.55
            ),
        })

        # t-SNE: subsample if needed
        g = torch.Generator().manual_seed(args.seed)
        def subsample(x):
            n = x.shape[0]
            if n <= args.max_points_tsne:
                return x
            idx = torch.randperm(n, generator=g)[:args.max_points_tsne]
            return x[idx]

        pos_s, neg_s = subsample(pos), subsample(neg)
        all_emb = torch.cat([pos_s, neg_s], dim=0).numpy()
        labels = np.array([1] * len(pos_s) + [0] * len(neg_s))

        # Projected embeddings (after removing first null direction)
        if directions:
            basis = _orthonormal_basis(directions[:1])
            pos_proj = _null_with_basis(pos_s, basis)
            neg_proj = _null_with_basis(neg_s, basis)
            all_proj = torch.cat([pos_proj, neg_proj], dim=0).numpy()
        else:
            all_proj = all_emb.copy()

        logger.info("Running t-SNE for %s ...", task["name"])
        orig_xy = tsne_2d(all_emb, seed=args.seed)
        proj_xy = tsne_2d(all_proj, seed=args.seed)

        tsne_data.append((task["name"], task["family"], orig_xy, proj_xy, labels, auc1, auc2))

    # Save INLP JSON
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"results": inlp_results}, indent=2))
    logger.info("Saved INLP results to %s", out_json)

    # Print table
    print(f"\n{'Task':<55} {'AUC iter1':>10} {'AUC iter2':>10} {'ΔAUC':>8} {'Rank-1?':>8}")
    print("-" * 95)
    for r in inlp_results:
        a1 = f"{r['iter1_auc']:.3f}" if r["iter1_auc"] is not None else "  -  "
        a2 = f"{r['iter2_auc']:.3f}" if r["iter2_auc"] is not None else "  -  "
        da = f"{r['delta_auc']:+.3f}" if r["delta_auc"] is not None else "  -  "
        ev = "yes" if r["rank1_linear_evidence"] else "no"
        print(f"{r['task']:<55} {a1:>10} {a2:>10} {da:>8} {ev:>8}")

    # t-SNE panel: rows = tasks, cols = [original, projected]
    n_tasks = len(tsne_data)
    fig, axes = plt.subplots(n_tasks, 2, figsize=(8, 3.2 * n_tasks), squeeze=False)
    fig.suptitle("INLP: Linear separability before and after null projection", fontsize=13, y=1.001)

    colors = {1: "#e05c5c", 0: "#5c8ae0"}
    alpha = 0.35
    s = 6

    for row, (name, family, orig_xy, proj_xy, labels, auc1, auc2) in enumerate(tsne_data):
        short_name = name.split(":")[-1] if ":" in name else name
        family_label = FAMILY_LABELS.get(family, family)

        for col, (xy, auc, subtitle) in enumerate([
            (orig_xy, auc1, "Original"),
            (proj_xy, auc2, "After null projection"),
        ]):
            ax = axes[row][col]
            for cls, label in [(1, "positive"), (0, "negative")]:
                mask = labels == cls
                ax.scatter(xy[mask, 0], xy[mask, 1], c=colors[cls], s=s, alpha=alpha, linewidths=0, label=label)
            auc_str = f"AUC={auc:.3f}" if auc is not None else "AUC=n/a"
            ax.set_title(f"{family_label} | {short_name}\n{subtitle} ({auc_str})", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0 and col == 1:
                ax.legend(fontsize=7, markerscale=2, loc="upper right")

    fig.tight_layout()
    out_plot = Path(args.output_plot)
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot, dpi=args.dpi, bbox_inches="tight")
    logger.info("Saved t-SNE panel to %s", out_plot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    main()
