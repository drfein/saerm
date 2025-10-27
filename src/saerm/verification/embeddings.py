from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, List

import torch


def verify_embeddings(
    payload: Dict[str, Any],
    *,
    expected_dim: Optional[int] = None,
    min_std: float = 1e-3,
    min_unique: int = 16,
    require_texts: bool = False,
    require_records: bool = False,
    expected_choices: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Lightweight sanity checks for cached embedding payloads."""

    issues = []

    if "embeddings" not in payload:
        return {"ok": False, "issues": ["payload missing 'embeddings' tensor"], "num_examples": 0, "embedding_dim": 0}

    embeddings = payload["embeddings"]
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.as_tensor(embeddings)

    if embeddings.ndim != 2:
        issues.append(f"expected 2D embeddings tensor, found shape {tuple(embeddings.shape)}")
        num_examples = embeddings.shape[0] if embeddings.ndim > 0 else 0
        embedding_dim = embeddings.shape[1] if embeddings.ndim > 1 else 0
        return {"ok": False, "issues": issues, "num_examples": int(num_examples), "embedding_dim": int(embedding_dim)}

    num_examples, embedding_dim = embeddings.shape

    if num_examples == 0:
        issues.append("no embeddings present")

    if expected_dim is not None and embedding_dim != expected_dim:
        issues.append(f"embedding dim {embedding_dim} != expected {expected_dim}")

    if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
        issues.append("found NaN/Inf values in embeddings")

    if num_examples > 0:
        std = embeddings.std(dim=0, unbiased=False)
        mean_std = float(std.mean().item())
        min_std_value = float(std.min().item())
        if mean_std < min_std:
            issues.append(f"mean per-dimension std {mean_std:.3e} < tolerance {min_std:.3e}")
        unique_rows = torch.unique(embeddings, dim=0).shape[0]
    else:
        mean_std = 0.0
        min_std_value = 0.0
        unique_rows = 0

    if min_unique and unique_rows < min(min_unique, max(1, num_examples)):
        issues.append(f"only {unique_rows} unique embedding rows detected")

    records = payload.get("records")
    texts: Optional[List[str]] = None
    if records is not None:
        if len(records) != num_examples:
            issues.append(f"records length {len(records)} != embeddings rows {num_examples}")
        texts = [str(entry.get("text", "")) for entry in records if entry.get("text") is not None]
        if require_records and not records:
            issues.append("payload missing 'records'")
        if expected_choices:
            expected_set = set(expected_choices)
            seen_by_example: Dict[Any, set] = {}
            for entry in records:
                example_index = entry.get("example_index")
                choice = entry.get("choice")
                if example_index is None:
                    issues.append("record missing example_index")
                    continue
                if choice is None:
                    issues.append(f"record for example {example_index} missing choice label")
                    continue
                seen_by_example.setdefault(example_index, set()).add(choice)
            for example_index, seen in seen_by_example.items():
                missing = expected_set - seen
                if missing:
                    issues.append(f"example {example_index} missing choices {sorted(missing)}")
    elif require_records:
        issues.append("payload missing 'records'")

    if require_texts:
        if texts is None:
            texts = payload.get("texts")
        if not texts or len(texts) != num_examples:
            issues.append("missing or mismatched texts accompanying embeddings")

    result = {
        "ok": not issues,
        "num_examples": int(num_examples),
        "embedding_dim": int(embedding_dim),
        "mean_std": mean_std,
        "min_std": min_std_value,
        "unique_examples": int(unique_rows),
        "issues": issues,
    }
    return result
