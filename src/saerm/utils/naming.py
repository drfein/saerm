from __future__ import annotations

import re
from typing import Iterable

_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


def slugify(parts: Iterable[str], max_length: int = 128) -> str:
    """Create a filesystem-friendly slug from the provided parts."""

    normalized = []
    for part in parts:
        lower = part.lower()
        cleaned = _SLUG_PATTERN.sub("-", lower).strip("-")
        if cleaned:
            normalized.append(cleaned)
    slug = "-".join(normalized)
    if len(slug) > max_length:
        slug = slug[:max_length]
    return slug
