#!/usr/bin/env python3
"""Evaluate a trained reward model head against a labeled dataset split."""

from __future__ import annotations

import argparse
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from saerm.config import HeadEvalConfig, load_experiment_config
from saerm.data import DatasetManager
from saerm.embeddings.cache import EmbeddingCacheManager
from saerm.heads import HeadFactory
from saerm.logging import configure_logging
from saerm.sae.inference import SAEFeatureExtractor
from saerm.storage import StorageManager


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Path to experiment config YAML")
    parser.add_argument(
        "--job-id",
        action="append",
        help="Head eval job id(s) to run; run all if omitted",
    )
    parser.add_argument("--dataset", help="Override dataset key defined in the head metadata")
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    parser.add_argument("--split", help="Override dataset split for evaluation")
    parser.add_argument("--embedding-job", help="Embedding job id to evaluate")
    parser.add_argument("--rejected-embedding-job", help="Optional rejected embedding job id for pairwise eval")
    args = parser.parse_args()

    configure_logging(args.log_level)
    config = load_experiment_config(args.config)
    storage = StorageManager(config.storage)
    datasets = DatasetManager(config)
    cache = EmbeddingCacheManager(storage)

    jobs = _select_eval_jobs(config.head_eval_jobs, args.job_id)
    if not jobs:
        logging.warning("No head eval jobs defined")
        return

    for job in jobs:
        head_job_id = job.head_job

        metadata_path = storage.head_metadata_path(head_job_id)
        metadata = storage.read_metadata(metadata_path)

        head_type = metadata["head_type"]
        model_path = storage.head_model_path(head_job_id)
        head = HeadFactory.load(head_type, str(model_path))

        dataset_key = args.dataset or job.dataset
        if dataset_key is None:
            raise SystemExit("Must provide --dataset or specify it in the eval job")

        split = args.split or job.split
        dataset = datasets.get(dataset_key, split)

        target_field = metadata.get("target_field")

        embedding_job_id = args.embedding_job or job.embedding_job
        if embedding_job_id is None:
            raise SystemExit("Must provide --embedding-job or specify it in the eval job")

        rejected_job = args.rejected_embedding_job or getattr(job, "rejected_embedding_job", None)

        if target_field:
            payload = cache.load_embeddings(embedding_job_id)
            params = metadata.get("params", {})
            supervised_choice = params.get("supervised_choice", "chosen")
            features_np, _ = _extract_features(payload, metadata, storage, choice=str(supervised_choice) if supervised_choice else None)
            targets = np.asarray(dataset[target_field][: features_np.shape[0]], dtype=np.float32)

            preds = head.predict(features_np)
            mse = float(np.mean((preds - targets) ** 2))
            corr = float(np.corrcoef(preds, targets)[0, 1]) if len(preds) > 1 else float("nan")

            logging.info("[%s] Evaluation results for head %s", job.job_id, head_job_id)
            logging.info("[%s] MSE: %.6f", job.job_id, mse)
            logging.info("[%s] Corr: %.6f", job.job_id, corr)
        else:
            chosen_payload = cache.load_embeddings(embedding_job_id)
            chosen_raw, chosen_ids = _extract_features(chosen_payload, metadata, storage, choice="chosen")

            if rejected_job:
                rejected_payload = cache.load_embeddings(rejected_job)
                rejected_raw, rejected_ids = _extract_features(rejected_payload, metadata, storage, choice="rejected")
            else:
                rejected_raw, rejected_ids = _extract_features(chosen_payload, metadata, storage, choice="rejected")

            # Strict per-example accuracy using all available candidates per example id
            chosen_scores_full = head.predict(chosen_raw)
            rejected_scores_full = head.predict(rejected_raw)
            strict_acc, strict_correct, strict_total = _strict_pairwise_accuracy(
                chosen_scores_full,
                chosen_ids,
                rejected_scores_full,
                rejected_ids,
            )

            # Pairwise Bradley-Terry metrics require aligned arrays
            chosen_np, rejected_np, aligned_ids = _align_features(chosen_raw, chosen_ids, rejected_raw, rejected_ids)
            if chosen_np.size == 0 or rejected_np.size == 0:
                raise ValueError(f"No overlapping samples available to evaluate head {head_job_id}")

            weights = _load_pairwise_weights(dataset, metadata.get("preference_weight_field"), aligned_ids, chosen_np.shape[0])

            weight_chosen = float(metadata.get("bt_weights", {}).get("chosen", 1.0))
            weight_rejected = float(metadata.get("bt_weights", {}).get("rejected", 1.0))

            chosen_scores = head.predict(chosen_np)
            rejected_scores = head.predict(rejected_np)
            bt_loss = _bt_loss(chosen_scores, rejected_scores, weights, weight_chosen, weight_rejected)
            bt_accuracy, correct_total, denom_total = _bt_accuracy_counts(
                chosen_scores,
                rejected_scores,
                weights,
                weight_chosen,
                weight_rejected,
            )
            margin = _bt_margin(chosen_scores, rejected_scores, weight_chosen, weight_rejected)

            logging.info("[%s] Pairwise evaluation results for head %s", job.job_id, head_job_id)
            logging.info("[%s] BT loss: %.6f", job.job_id, bt_loss)
            if weights is None:
                logging.info("[%s] BT accuracy: %.6f (%d/%d)", job.job_id, bt_accuracy, int(correct_total), int(denom_total))
            else:
                logging.info("[%s] BT accuracy: %.6f (%.2f/%.2f weighted)", job.job_id, bt_accuracy, correct_total, denom_total)
            logging.info("[%s] Margin mean: %.6f", job.job_id, margin)
            logging.info("[%s] Strict per-example accuracy: %.6f (%d/%d)", job.job_id, strict_acc, strict_correct, strict_total)
            _report_subset_metrics(
                dataset,
                aligned_ids,
                chosen_scores,
                rejected_scores,
                weights,
                weight_chosen,
                weight_rejected,
            )


def _select_eval_jobs(jobs: List[HeadEvalConfig], requested_ids: Optional[List[str]]) -> List[HeadEvalConfig]:
    if not requested_ids:
        return list(jobs)
    job_map = {job.job_id: job for job in jobs}
    missing = [job_id for job_id in requested_ids if job_id not in job_map]
    if missing:
        raise SystemExit(f"Unknown head eval job id(s): {', '.join(missing)}")
    return [job_map[job_id] for job_id in requested_ids]


def _build_sae_extractor(storage: StorageManager, metadata: dict, input_dim: int) -> Optional[SAEFeatureExtractor]:
    if not metadata.get("sae_job"):
        return None
    sae_meta = storage.read_metadata(storage.sae_metadata_path(metadata["sae_job"]))
    return SAEFeatureExtractor(
        checkpoint_path=str(storage.sae_checkpoint_path(metadata["sae_job"])),
        input_dim=sae_meta.get("input_dim", input_dim),
        hidden_dim=sae_meta["hidden_size"],
        k_active=sae_meta.get("k_active"),
        device="cpu",
    )


def _bt_loss(
    chosen_scores: np.ndarray,
    rejected_scores: np.ndarray,
    sample_weights: Optional[np.ndarray],
    weight_chosen: float,
    weight_rejected: float,
) -> float:
    diff = weight_chosen * chosen_scores - weight_rejected * rejected_scores
    losses = np.logaddexp(0.0, -diff)
    if sample_weights is not None and sample_weights.size > 0:
        total = float(np.sum(sample_weights))
        if total > 0.0:
            return float(np.sum(losses * sample_weights) / total)
    return float(np.mean(losses))


def _bt_accuracy(
    chosen_scores: np.ndarray,
    rejected_scores: np.ndarray,
    sample_weights: Optional[np.ndarray],
    weight_chosen: float,
    weight_rejected: float,
) -> float:
    diff = weight_chosen * chosen_scores - weight_rejected * rejected_scores
    correct = diff > 0.0
    if sample_weights is not None and sample_weights.size > 0:
        total = float(np.sum(sample_weights))
        if total > 0.0:
            return float(np.sum(correct * sample_weights) / total)
    return float(np.mean(correct))


def _bt_accuracy_counts(
    chosen_scores: np.ndarray,
    rejected_scores: np.ndarray,
    sample_weights: Optional[np.ndarray],
    weight_chosen: float,
    weight_rejected: float,
) -> Tuple[float, float, float]:
    diff = weight_chosen * chosen_scores - weight_rejected * rejected_scores
    correct = diff > 0.0
    if sample_weights is not None and sample_weights.size > 0:
        weights = np.asarray(sample_weights, dtype=np.float64)
        total = float(weights.sum())
        if total <= 0.0:
            total = float(correct.size)
            correct_total = float(correct.sum())
        else:
            correct_total = float((correct * weights).sum())
        accuracy = correct_total / total if total > 0 else 0.0
        return accuracy, correct_total, total
    total = float(correct.size)
    correct_total = float(correct.sum())
    accuracy = correct_total / total if total > 0 else 0.0
    return accuracy, correct_total, total


def _bt_margin(
    chosen_scores: np.ndarray,
    rejected_scores: np.ndarray,
    weight_chosen: float,
    weight_rejected: float,
) -> float:
    diff = weight_chosen * chosen_scores - weight_rejected * rejected_scores
    return float(np.mean(diff))


def _report_subset_metrics(
    dataset,
    example_ids: Optional[np.ndarray],
    chosen_scores: np.ndarray,
    rejected_scores: np.ndarray,
    sample_weights: Optional[np.ndarray],
    weight_chosen: float,
    weight_rejected: float,
) -> None:
    if example_ids is None:
        logging.info("Skipping subset breakdown: example ids unavailable")
        return
    subset_column = _get_column(dataset, "subset")
    if subset_column is None:
        logging.info("Skipping subset breakdown: dataset missing 'subset' field")
        return
    diff = weight_chosen * chosen_scores - weight_rejected * rejected_scores
    correct = diff > 0.0
    if sample_weights is not None and sample_weights.size > 0:
        weights = np.asarray(sample_weights, dtype=np.float64)
        weighted = True
    else:
        weights = np.ones_like(correct, dtype=np.float64)
        weighted = False
    summary: Dict[str, Tuple[float, float]] = {}
    ids = example_ids.astype(np.int64, copy=False)
    for idx, example_id in enumerate(ids):
        if example_id < 0 or example_id >= len(subset_column):
            continue
        subset_key = str(subset_column[example_id])
        total, hits = summary.get(subset_key, (0.0, 0.0))
        weight = float(weights[idx])
        summary[subset_key] = (total + weight, hits + (weight if correct[idx] else 0.0))
    if not summary:
        logging.info("Subset breakdown produced no entries (check dataset alignment)")
        return
    logging.info("Subset breakdown:")
    for subset_key in sorted(summary.keys()):
        total, hits = summary[subset_key]
        acc = hits / total if total > 0 else 0.0
        if weighted:
            logging.info("  %s: accuracy %.6f (%.2f/%.2f weighted)", subset_key, acc, hits, total)
        else:
            logging.info("  %s: accuracy %.6f (%d/%d)", subset_key, acc, int(hits), int(total))


def _extract_features(
    payload: Dict[str, Any],
    metadata: dict,
    storage: StorageManager,
    *,
    choice: Optional[str],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    embeddings = payload.get("embeddings")
    if embeddings is None:
        raise KeyError("Embedding payload missing 'embeddings'")
    embeddings = embeddings.float()
    records = payload.get("records")
    indices, example_ids = _select_indices(records, embeddings.shape[0], choice)
    if not indices:
        raise ValueError(f"No embeddings found for choice {choice!r}")
    index_tensor = torch.tensor(indices, dtype=torch.long, device=embeddings.device)
    selected = embeddings.index_select(0, index_tensor)
    extractor = _build_sae_extractor(storage, metadata, selected.shape[1])
    if extractor:
        features = extractor.transform(selected)
    else:
        features = selected
    return features.cpu().numpy().astype(np.float32, copy=False), example_ids


def _select_indices(
    records: Optional[List[Dict[str, Any]]],
    total_rows: int,
    choice: Optional[str],
) -> Tuple[List[int], Optional[np.ndarray]]:
    if records and len(records) == total_rows:
        if choice is None:
            indices = list(range(total_rows))
        else:
            indices = [idx for idx, entry in enumerate(records) if entry.get("choice") == choice]
        example_ids = np.asarray(
            [int(records[idx].get("example_index", idx)) for idx in indices],
            dtype=np.int64,
        ) if indices else None
        return indices, example_ids

    logging.warning(
        "Embedding payload missing aligned records; falling back to alternating split for choice %s",
        choice or "all",
    )
    if choice == "rejected":
        indices = list(range(1, total_rows, 2))
    elif choice == "chosen":
        indices = list(range(0, total_rows, 2))
    else:
        indices = list(range(total_rows))
    example_ids = np.arange(len(indices), dtype=np.int64) if indices else None
    return indices, example_ids


def _align_features(
    chosen_features: np.ndarray,
    chosen_ids: Optional[np.ndarray],
    rejected_features: np.ndarray,
    rejected_ids: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if chosen_ids is None or rejected_ids is None:
        size = min(len(chosen_features), len(rejected_features))
        if len(chosen_features) != len(rejected_features):
            logging.warning(
                "Feature counts differ (%s vs %s); truncating to %s",
                len(chosen_features),
                len(rejected_features),
                size,
            )
        return chosen_features[:size], rejected_features[:size], None

    chosen_ids = chosen_ids.astype(np.int64, copy=False)
    rejected_ids = rejected_ids.astype(np.int64, copy=False)
    chosen_ids = chosen_ids[: chosen_features.shape[0]]
    rejected_ids = rejected_ids[: rejected_features.shape[0]]

    if np.array_equal(chosen_ids, rejected_ids):
        size = min(chosen_features.shape[0], rejected_features.shape[0])
        return chosen_features[:size], rejected_features[:size], chosen_ids[:size]

    chosen_lookup = {int(idx): pos for pos, idx in enumerate(chosen_ids)}
    alignment = [(chosen_lookup[int(idx)], pos) for pos, idx in enumerate(rejected_ids) if int(idx) in chosen_lookup]
    if not alignment:
        raise ValueError("No overlapping example ids between chosen and rejected embeddings")

    chosen_positions = np.asarray([item[0] for item in alignment], dtype=np.int64)
    rejected_positions = np.asarray([item[1] for item in alignment], dtype=np.int64)
    aligned_ids = chosen_ids[chosen_positions]

    if len(alignment) < min(chosen_features.shape[0], rejected_features.shape[0]):
        logging.warning(
            "Only %s aligned examples between chosen (%s) and rejected (%s)",
            len(alignment),
            chosen_features.shape[0],
            rejected_features.shape[0],
        )

    return chosen_features[chosen_positions], rejected_features[rejected_positions], aligned_ids


def _strict_pairwise_accuracy(
    chosen_scores: np.ndarray,
    chosen_ids: Optional[np.ndarray],
    rejected_scores: np.ndarray,
    rejected_ids: Optional[np.ndarray],
) -> Tuple[float, int, int]:
    if chosen_ids is None or rejected_ids is None:
        size = min(len(chosen_scores), len(rejected_scores))
        if size == 0:
            return 0.0, 0, 0
        correct = int(np.sum(chosen_scores[:size] > rejected_scores[:size]))
        return float(correct / size), correct, size

    chosen_ids = chosen_ids.astype(np.int64, copy=False)
    rejected_ids = rejected_ids.astype(np.int64, copy=False)
    chosen_map: Dict[int, List[float]] = {}
    for score, idx in zip(chosen_scores, chosen_ids):
        chosen_map.setdefault(int(idx), []).append(float(score))
    rejected_map: Dict[int, List[float]] = {}
    for score, idx in zip(rejected_scores, rejected_ids):
        rejected_map.setdefault(int(idx), []).append(float(score))

    ids = sorted(set(chosen_map.keys()) & set(rejected_map.keys()))
    if not ids:
        return 0.0, 0, 0
    correct = 0
    for example_id in ids:
        c_scores = chosen_map.get(example_id, [])
        r_scores = rejected_map.get(example_id, [])
        if not c_scores or not r_scores:
            continue
        if min(c_scores) > max(r_scores):
            correct += 1
    total = len(ids)
    acc = float(correct / total) if total > 0 else 0.0
    return acc, correct, total


def _get_column(dataset, field: str):
    if hasattr(dataset, "column_names") and field not in dataset.column_names:
        return None
    try:
        return dataset[field]
    except Exception:
        return None


def _load_pairwise_weights(
    dataset,
    weight_field: Optional[str],
    example_ids: Optional[np.ndarray],
    fallback_size: int,
) -> Optional[np.ndarray]:
    if not weight_field:
        return None
    if hasattr(dataset, "column_names") and weight_field not in dataset.column_names:
        logging.warning("Weight field '%s' missing in dataset; ignoring weights", weight_field)
        return None
    column = dataset[weight_field]
    if example_ids is not None and example_ids.size > 0:

        max_id = int(example_ids.max())
        if len(column) <= max_id:
            logging.warning(
                "Weight column shorter (%s) than highest example id (%s); truncating",
                len(column),
                max_id,
            )
        values = [column[i] for i in example_ids if i < len(column)]
    else:
        if len(column) < fallback_size:
            logging.warning("Weight column shorter (%s) than features (%s); truncating", len(column), fallback_size)
        values = column[:fallback_size]
    weights = np.asarray(values, dtype=np.float32)
    if weights.size == 0:
        return None
    return np.clip(weights, a_min=0.0, a_max=None)


if __name__ == "__main__":
    main()
