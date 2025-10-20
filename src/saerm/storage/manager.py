from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

from ..config import StorageConfig
from ..utils.naming import slugify


class StorageManager:
    """Utility for managing experiment artifacts on disk."""

    def __init__(self, config: StorageConfig) -> None:
        self._config = config
        self._root = config.root_path()

    @property
    def root(self) -> Path:
        return self._root

    def prepare(self) -> None:
        for subdir in (self._config.embeddings_dir, self._config.sae_dir, self._config.heads_dir):
            (self.root / subdir).mkdir(parents=True, exist_ok=True)

    # Embeddings -----------------------------------------------------------------

    def embedding_dir(self, job_id: str) -> Path:
        return self._job_dir(self._config.embeddings_dir, job_id)

    def embedding_tensor_path(self, job_id: str) -> Path:
        return self.embedding_dir(job_id) / "embeddings.pt"

    def embedding_metadata_path(self, job_id: str) -> Path:
        return self.embedding_dir(job_id) / "metadata.json"

    # SAE ------------------------------------------------------------------------

    def sae_dir(self, job_id: str) -> Path:
        return self._job_dir(self._config.sae_dir, job_id)

    def sae_checkpoint_path(self, job_id: str) -> Path:
        return self.sae_dir(job_id) / "checkpoint.pt"

    def sae_metadata_path(self, job_id: str) -> Path:
        return self.sae_dir(job_id) / "metadata.json"

    # Heads ----------------------------------------------------------------------

    def head_dir(self, job_id: str) -> Path:
        return self._job_dir(self._config.heads_dir, job_id)

    def head_model_path(self, job_id: str) -> Path:
        return self.head_dir(job_id) / "model.bin"

    def head_metadata_path(self, job_id: str) -> Path:
        return self.head_dir(job_id) / "metadata.json"

    # Generic helpers ------------------------------------------------------------

    def write_metadata(self, path: Path, payload: Dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    def read_metadata(self, path: Path) -> Dict:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _job_dir(self, category: str, job_id: str) -> Path:
        slug = slugify([job_id])
        path = self.root / category / slug
        path.mkdir(parents=True, exist_ok=True)
        return path


def infer_job_id(parts: Iterable[str]) -> str:
    return slugify(parts)
