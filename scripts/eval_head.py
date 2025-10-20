#!/usr/bin/env python3
"""Evaluate a trained reward model head against a labeled dataset split."""

from __future__ import annotations

import argparse
import logging

import numpy as np

from saerm.config import load_experiment_config
from saerm.data import DatasetManager
from saerm.embeddings.cache import EmbeddingCacheManager
from saerm.heads import HeadFactory
from saerm.logging import configure_logging
from saerm.sae.inference import SAEFeatureExtractor
from saerm.storage import StorageManager


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Path to experiment config YAML")
    parser.add_argument("--job-id", required=True, help="Head job id to evaluate")
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    parser.add_argument("--split", help="Override dataset split for evaluation")
    args = parser.parse_args()

    configure_logging(args.log_level)
    config = load_experiment_config(args.config)
    storage = StorageManager(config.storage)
    datasets = DatasetManager(config)
    cache = EmbeddingCacheManager(storage)

    metadata_path = storage.head_metadata_path(args.job_id)
    metadata = storage.read_metadata(metadata_path)

    head_type = metadata["head_type"]
    model_path = storage.head_model_path(args.job_id)
    head = HeadFactory.load(head_type, str(model_path))

    embeddings = cache.load_embeddings(metadata["embedding_job"])["embeddings"].float()
    features = embeddings
    if metadata.get("sae_job"):
        sae_meta = storage.read_metadata(storage.sae_metadata_path(metadata["sae_job"]))
        extractor = SAEFeatureExtractor(
            checkpoint_path=str(storage.sae_checkpoint_path(metadata["sae_job"])),
            input_dim=embeddings.shape[1],
            hidden_dim=sae_meta["hidden_size"],
            device="cpu",
        )
        features = extractor.transform(embeddings)

    dataset_key = metadata["dataset"]
    split = args.split or metadata.get("eval_split") or metadata.get("train_split")
    dataset = datasets.get(dataset_key, split)
    target_field = metadata["target_field"]
    targets = np.asarray(dataset[target_field][: features.shape[0]], dtype=np.float32)

    preds = head.predict(features.numpy())
    mse = float(np.mean((preds - targets) ** 2))
    corr = float(np.corrcoef(preds, targets)[0, 1]) if len(preds) > 1 else float("nan")

    logging.info("Evaluation results for %s", args.job_id)
    logging.info("MSE: %.6f", mse)
    logging.info("Corr: %.6f", corr)


if __name__ == "__main__":
    main()
