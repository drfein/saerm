#!/usr/bin/env python3
"""Train reward model heads on top of cached features."""

from __future__ import annotations

import argparse
import logging
from typing import Iterable, List

from saerm.config import HeadTrainingConfig, load_experiment_config
from saerm.data import DatasetManager
from saerm.embeddings.cache import EmbeddingCacheManager
from saerm.heads.trainer import HeadTrainer
from saerm.logging import configure_logging
from saerm.storage import StorageManager


def _select_jobs(jobs: Iterable[HeadTrainingConfig], requested_ids: List[str] | None) -> List[HeadTrainingConfig]:
    if not requested_ids:
        return list(jobs)
    job_map = {job.job_id: job for job in jobs}
    missing = [job_id for job_id in requested_ids if job_id not in job_map]
    if missing:
        raise SystemExit(f"Unknown head job id(s): {', '.join(missing)}")
    return [job_map[job_id] for job_id in requested_ids]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Path to experiment config YAML")
    parser.add_argument("--job-id", action="append", help="Head job id(s) to train")
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    args = parser.parse_args()

    configure_logging(args.log_level)
    config = load_experiment_config(args.config)
    storage = StorageManager(config.storage)
    storage.prepare()
    cache = EmbeddingCacheManager(storage)
    datasets = DatasetManager(config)

    jobs = _select_jobs(config.head_jobs, args.job_id)
    if not jobs:
        logging.warning("No head jobs defined")
        return

    for job in jobs:
        logging.info("Training head job %s (%s)", job.job_id, job.head_type)
        trainer = HeadTrainer(storage, cache, datasets, job)
        trainer.train()


if __name__ == "__main__":
    main()
