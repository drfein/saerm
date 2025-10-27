#!/usr/bin/env python3
"""Cache transformer embeddings for downstream SAE/RM training."""

from __future__ import annotations

import argparse
import logging
from typing import Iterable, List

from saerm.config import EmbeddingJobConfig, load_experiment_config
from saerm.data import DatasetManager
from saerm.embeddings.cache import EmbeddingCacheManager
from saerm.logging import configure_logging
from saerm.storage import StorageManager


def _select_jobs(jobs: Iterable[EmbeddingJobConfig], requested_ids: List[str] | None) -> List[EmbeddingJobConfig]:
    if not requested_ids:
        return list(jobs)
    job_map = {job.job_id: job for job in jobs}
    missing = [job_id for job_id in requested_ids if job_id not in job_map]
    if missing:
        raise SystemExit(f"Unknown embedding job id(s): {', '.join(missing)}")
    return [job_map[job_id] for job_id in requested_ids]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to experiment config YAML")
    parser.add_argument("--job-id", action="append", help="Embedding job id(s) to run (repeatable)")
    parser.add_argument("--log-level", type=str, default="INFO", help="Python logging level")
    args = parser.parse_args()

    configure_logging(args.log_level)
    logging.info("Loading configuration from %s", args.config)
    config = load_experiment_config(args.config)
    storage = StorageManager(config.storage)
    storage.prepare()
    datasets = DatasetManager(config)
    cache = EmbeddingCacheManager(storage)

    jobs = _select_jobs(config.embedding_jobs, args.job_id)
    if not jobs:
        logging.warning("No embedding jobs defined; exiting")
        return

    for job in jobs:
        logging.info("Running embedding job %s (%s)", job.job_id, job.model)
        dataset = datasets.get(job.dataset, None)
        cache.run_job(job, dataset)


if __name__ == "__main__":
    main()
