from __future__ import annotations

import logging
from typing import Optional


def configure_logging(level: str = "INFO", freeze_existing: bool = False) -> None:
    if freeze_existing and logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
