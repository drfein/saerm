#!/usr/bin/env python3
"""Baseline: run the configured reward model in sequence classification mode on an eval dataset and report accuracy."""

from __future__ import annotations

import argparse
import logging
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
try:  # pragma: no cover - optional dependency
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

from saerm.config import EmbeddingJobConfig, HeadEvalConfig, load_experiment_config
from saerm.data.datasets import DatasetManager
from saerm.embeddings.rendering import ensure_chat_messages, render_messages
from saerm.logging import configure_logging


def _select_eval_jobs(jobs: Iterable[HeadEvalConfig], requested_ids: Optional[List[str]]) -> List[HeadEvalConfig]:
    if not requested_ids:
        return list(jobs)
    job_map = {job.job_id: job for job in jobs}
    missing = [job_id for job_id in requested_ids if job_id not in job_map]
    if missing:
        raise SystemExit(f"Unknown head eval job id(s): {', '.join(missing)}")
    return [job_map[job_id] for job_id in requested_ids]


def _scores_from_logits(logits: torch.Tensor) -> torch.Tensor:
    # Supports shapes: (B, 1) or (B, 2+) classification logits
    if logits.ndim != 2:
        raise ValueError(f"Expected logits of shape (B, C), found {tuple(logits.shape)}")
    if logits.shape[1] == 1:
        return logits[:, 0]
    if logits.shape[1] == 2:
        # Use logit difference which is invariant to label id choice
        return logits[:, 1] - logits[:, 0]
    # Fall back to the max-class logit (monotonic with confidence)
    return torch.max(logits, dim=1).values


def _is_chat_messages(value: object) -> bool:
    return isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict) and "role" in value[0]


def _render_candidates(
    value: object,
    prompt_value: Optional[str],
    tokenizer: AutoTokenizer,
) -> List[str]:
    # Treat a list of plain strings as multiple candidate responses; otherwise single candidate
    if isinstance(value, list) and not _is_chat_messages(value):
        texts: List[str] = []
        for item in value:
            messages = ensure_chat_messages(item, prompt_value)
            texts.append(render_messages(messages, tokenizer))
        return texts
    messages = ensure_chat_messages(value, prompt_value)
    return [render_messages(messages, tokenizer)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Path to experiment config YAML")
    parser.add_argument(
        "--job-id",
        action="append",
        help="Head eval job id(s) to run; run all if omitted",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for tokenization/inference (overrides config)")
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    args = parser.parse_args()

    configure_logging(args.log_level)
    config = load_experiment_config(args.config)
    dataset_manager = DatasetManager(config)

    eval_jobs = _select_eval_jobs(config.head_eval_jobs, args.job_id)
    if not eval_jobs:
        logging.warning("No head eval jobs defined")
        return

    embedding_job_map = {job.job_id: job for job in config.embedding_jobs}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    default_batch_size = config.baseline.batch_size if getattr(config, "baseline", None) else 16

    for eval_job in eval_jobs:
        if eval_job.embedding_job not in embedding_job_map:
            raise SystemExit(f"Embedding job '{eval_job.embedding_job}' not found for eval job '{eval_job.job_id}'")
        embed_job: EmbeddingJobConfig = embedding_job_map[eval_job.embedding_job]

        dataset = dataset_manager.get(eval_job.dataset, eval_job.split)
        if not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__"):
            raise SystemExit(f"Dataset '{eval_job.dataset}' does not support random access; cannot run baseline eval")

        logging.info("[%s] Loading model %s", eval_job.job_id, embed_job.model)
        tokenizer = AutoTokenizer.from_pretrained(embed_job.tokenizer or embed_job.model)
        model = AutoModelForSequenceClassification.from_pretrained(embed_job.model)
        # Ensure a padding token is defined for batched inference
        if getattr(tokenizer, "pad_token_id", None) is None:
            if getattr(tokenizer, "eos_token_id", None) is not None:
                tokenizer.pad_token = tokenizer.eos_token
            elif getattr(tokenizer, "unk_token", None) is not None:
                tokenizer.pad_token = tokenizer.unk_token
            else:
                tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
                try:
                    model.resize_token_embeddings(len(tokenizer))
                except Exception:
                    pass
        if getattr(model.config, "pad_token_id", None) is None and getattr(tokenizer, "pad_token_id", None) is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
        model.to(device)
        model.eval()

        prompt_field = embed_job.prompt_field or "prompt"
        chosen_field = embed_job.chosen_field or "chosen"
        rejected_field = embed_job.rejected_field or "rejected"

        total = 0
        correct = 0

        batch_size = max(1, (args.batch_size if args.batch_size is not None else default_batch_size))
        num_examples = len(dataset)
        iterator = range(num_examples)
        if tqdm is not None:
            iterator = tqdm(iterator, desc=f"baseline:{eval_job.job_id}", total=num_examples)
        for idx in iterator:
            record = dataset[idx]
            prompt_value = record.get(prompt_field)
            chosen_candidates = _render_candidates(record[chosen_field], prompt_value, tokenizer)
            rejected_candidates = _render_candidates(record[rejected_field], prompt_value, tokenizer)

            with torch.no_grad():
                # Score all chosen candidates
                if chosen_candidates:
                    tk_chosen = tokenizer(
                        chosen_candidates,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    ).to(device)
                    out_chosen = model(**tk_chosen)
                    scores_chosen = _scores_from_logits(out_chosen.logits).detach().float()
                else:
                    scores_chosen = torch.empty(0, dtype=torch.float32, device=device)

                # Score all rejected candidates
                if rejected_candidates:
                    tk_rejected = tokenizer(
                        rejected_candidates,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    ).to(device)
                    out_rejected = model(**tk_rejected)
                    scores_rejected = _scores_from_logits(out_rejected.logits).detach().float()
                else:
                    scores_rejected = torch.empty(0, dtype=torch.float32, device=device)

            total += 1
            if scores_chosen.numel() == 0 or scores_rejected.numel() == 0:
                # If either side missing, cannot verify; count as incorrect
                continue
            if torch.min(scores_chosen) > torch.max(scores_rejected):
                correct += 1

        accuracy = (correct / total) if total > 0 else 0.0
        logging.info("[%s] Baseline accuracy: %.6f (%d/%d)", eval_job.job_id, accuracy, correct, total)


if __name__ == "__main__":
    main()


