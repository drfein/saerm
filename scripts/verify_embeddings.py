#!/usr/bin/env python3
"""Verify cached embeddings for jobs defined in the experiment config."""

from __future__ import annotations

import argparse
import sys

from saerm.config import load_experiment_config
from saerm.embeddings.cache import EmbeddingCacheManager
from saerm.storage import StorageManager
from saerm.verification import verify_embeddings


def _verify_job(storage: StorageManager, manager: EmbeddingCacheManager, job_id: str) -> int:
    try:
        payload = manager.load_embeddings(job_id)
    except FileNotFoundError:
        print(f"[ERROR] Embedding tensor for job '{job_id}' not found", file=sys.stderr)
        return 1
    try:
        metadata = manager.load_metadata(job_id)
    except FileNotFoundError:
        metadata = {}

    expected_dim = int(metadata.get("embedding_dim") or 0) or None
    result = verify_embeddings(
        payload,
        expected_dim=expected_dim,
        require_texts=True,
        require_records=True,
        expected_choices=("chosen", "rejected"),
    )

    if result["ok"]:
        print(
            f"[OK] {job_id}: examples={result['num_examples']} dim={result['embedding_dim']} "
            f"mean_std={result['mean_std']:.3e} unique={result['unique_examples']}"
        )
        return 0

    print(f"[WARN] {job_id}: detected {len(result['issues'])} issue(s)")
    for issue in result["issues"]:
        print(f"  - {issue}")
    print(
        f"    stats: examples={result['num_examples']} dim={result['embedding_dim']} "
        f"mean_std={result['mean_std']:.3e} min_std={result['min_std']:.3e} unique={result['unique_examples']}"
    )
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Path to experiment config YAML")
    parser.add_argument("--job-id", action="append", help="Specific embedding job id(s) to verify")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    storage = StorageManager(config.storage)
    manager = EmbeddingCacheManager(storage)

    job_ids = args.job_id or [job.job_id for job in config.embedding_jobs]
    if not job_ids:
        print("No embedding jobs defined in config.", file=sys.stderr)
        raise SystemExit(1)

    status = 0
    for job_id in job_ids:
        status |= _verify_job(storage, manager, job_id)
    raise SystemExit(status)


if __name__ == "__main__":
    main()
