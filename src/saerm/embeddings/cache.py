from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional

import torch
from datasets import Dataset, IterableDataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from ..config import EmbeddingJobConfig
from ..storage import StorageManager

logger = logging.getLogger(__name__)


def _parse_layer_index(layer_spec: str | int, total_layers: int) -> int:
    if isinstance(layer_spec, int):
        if layer_spec >= 0:
            return layer_spec
        return total_layers + layer_spec
    if layer_spec.isdigit():
        return int(layer_spec)
    if layer_spec.startswith("-") and layer_spec[1:].isdigit():
        return total_layers + int(layer_spec)
    if layer_spec == "last":
        return total_layers - 1
    raise ValueError(f"Unsupported layer specification: {layer_spec}")


class EmbeddingCacheManager:
    """Create and persist embeddings for downstream SAE training."""

    def __init__(self, storage: StorageManager, device: Optional[str] = None) -> None:
        self._storage = storage
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def run_job(self, job: EmbeddingJobConfig, dataset: Dataset | IterableDataset) -> None:
        logger.info("Running embedding job %s", job.job_id)
        model = AutoModel.from_pretrained(job.model, output_hidden_states=True)
        model.to(self._device)
        model.eval()
        tokenizer_name = job.tokenizer or job.model
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        storage_dir = self._storage.embedding_dir(job.job_id)
        tensor_path = self._storage.embedding_tensor_path(job.job_id)
        metadata_path = self._storage.embedding_metadata_path(job.job_id)

        embeddings: List[torch.Tensor] = []
        records: List[Dict] = []
        total_processed = 0

        iterable: Iterable
        if isinstance(dataset, Dataset):
            iterable = dataset
        else:
            iterable = dataset

        for idx, record in enumerate(tqdm(iterable, desc=f"embeddings:{job.job_id}")):
            if job.max_examples is not None and idx >= job.max_examples:
                break
            payload = record[job.prompt_field]
            if isinstance(payload, (list, tuple)):
                payload = "\n".join(payload)
            inputs = tokenizer(payload, return_tensors="pt", truncation=True)
            inputs = {name: tensor.to(self._device) for name, tensor in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise RuntimeError("Model did not return hidden states; set output_hidden_states=True")
            layer_idx = _parse_layer_index(job.layer, len(hidden_states))
            # Use CLS token representation or mean pool if CLS not available.
            layer_tensor = hidden_states[layer_idx]
            if layer_tensor.ndim == 3:
                cls_embedding = layer_tensor[:, 0]
            else:
                cls_embedding = layer_tensor
            embeddings.append(cls_embedding.cpu())
            records.append({
                "index": idx,
                "text_length": len(payload) if isinstance(payload, str) else 0,
            })
            total_processed += 1

        if not embeddings:
            raise ValueError(f"No embeddings generated for job {job.job_id}")

        combined = torch.cat(embeddings, dim=0)
        torch.save({"embeddings": combined}, tensor_path)
        logger.info("Saved embeddings to %s", tensor_path)

        self._storage.write_metadata(metadata_path, {
            "job_id": job.job_id,
            "model": job.model,
            "layer": job.layer,
            "dataset": job.dataset,
            "count": total_processed,
            "device": self._device,
        })

    def load_embeddings(self, job_id: str) -> Dict[str, torch.Tensor]:
        tensor_path = self._storage.embedding_tensor_path(job_id)
        return torch.load(tensor_path, map_location="cpu")

    def load_metadata(self, job_id: str) -> Dict:
        metadata_path = self._storage.embedding_metadata_path(job_id)
        return self._storage.read_metadata(metadata_path)
