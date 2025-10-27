#!/usr/bin/env python3
"""Run configured AutoInterp jobs to score SAE features with an LLM."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from typing import Iterable, List

from saerm.autointerp import AutoInterpreter
from saerm.config import AutoInterpJobConfig, load_experiment_config
from saerm.data.datasets import DatasetManager
from saerm.logging import configure_logging
from saerm.storage import StorageManager


def _select_jobs(
    jobs: Iterable[AutoInterpJobConfig], requested_ids: List[str] | None
) -> List[AutoInterpJobConfig]:
    if not requested_ids:
        return list(jobs)
    job_map = {job.job_id: job for job in jobs}
    missing = [job_id for job_id in requested_ids if job_id not in job_map]
    if missing:
        raise SystemExit(f"Unknown AutoInterp job id(s): {', '.join(missing)}")
    return [job_map[job_id] for job_id in requested_ids]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Path to experiment config YAML")
    parser.add_argument(
        "--job-id",
        action="append",
        help="AutoInterp job id(s) to run; run all if omitted",
    )
    parser.add_argument(
        "--head-job",
        help="Optionally override AutoInterp feature selection using the specified head job",
    )
    parser.add_argument(
        "--head-top-n",
        type=int,
        default=None,
        help="Number of most positive/negative head weights to interpret (per sign)",
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    args = parser.parse_args()

    configure_logging(args.log_level)
    config = load_experiment_config(args.config)
    storage = StorageManager(config.storage)
    storage.prepare()
    dataset_manager = DatasetManager(config)

    jobs = _select_jobs(config.autointerp_jobs, args.job_id)
    if not jobs:
        logging.warning("No AutoInterp jobs defined")
        return

    for job in jobs:
        if args.head_job:
            kwargs = {"head_job": args.head_job}
            if args.head_top_n is not None:
                kwargs["head_top_n"] = args.head_top_n
            job = replace(job, **kwargs)
        logging.info("Running AutoInterp job %s for SAE %s", job.job_id, job.sae_job)
        interpreter = AutoInterpreter(
            storage,
            job,
            config=config,
            dataset_manager=dataset_manager,
        )
        interpreter.run()


if __name__ == "__main__":
    main()
