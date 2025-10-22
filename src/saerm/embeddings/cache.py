from __future__ import annotations

import logging
import json
from typing import Any, Dict, Iterable, List, Optional, Sequence

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
        paired_embeddings: List[torch.Tensor] | None = [] if job.paired_chat_messages_field or job.paired_text_field else None
        total_processed = 0
        chat_template_used = False
        paired_chat_template_used = False
        example_ids: List[int] = []

        iterable: Iterable
        if isinstance(dataset, Dataset):
            iterable = dataset
        else:
            iterable = dataset

        for idx, record in enumerate(tqdm(iterable, desc=f"embeddings:{job.job_id}")):
            if job.max_examples is not None and idx >= job.max_examples:
                break
            payload_text, used_template = self._build_input_text(record, job, tokenizer)
            chat_template_used = chat_template_used or used_template
            inputs = tokenizer(payload_text, return_tensors="pt", truncation=True)
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

            if paired_embeddings is not None:
                paired_text, paired_template = self._build_input_text(
                    record,
                    job,
                    tokenizer,
                    chat_field=job.paired_chat_messages_field,
                    text_field=job.paired_text_field,
                )
                paired_chat_template_used = paired_chat_template_used or paired_template
                paired_inputs = tokenizer(paired_text, return_tensors="pt", truncation=True)
                paired_inputs = {name: tensor.to(self._device) for name, tensor in paired_inputs.items()}
                with torch.no_grad():
                    paired_outputs = model(**paired_inputs)
                paired_hidden_states = paired_outputs.hidden_states
                if paired_hidden_states is None:
                    raise RuntimeError("Model did not return hidden states; set output_hidden_states=True")
                paired_layer_tensor = paired_hidden_states[layer_idx]
                if paired_layer_tensor.ndim == 3:
                    paired_cls = paired_layer_tensor[:, 0]
                else:
                    paired_cls = paired_layer_tensor
                paired_embeddings.append(paired_cls.cpu())

            example_ids.append(idx)
            total_processed += 1

        if not embeddings:
            raise ValueError(f"No embeddings generated for job {job.job_id}")

        combined = torch.cat(embeddings, dim=0)
        payload: Dict[str, Any] = {"embeddings": combined, "example_ids": list(example_ids)}
        if paired_embeddings:
            paired_combined = torch.cat(paired_embeddings, dim=0)
            payload["embeddings_paired"] = paired_combined
            payload["paired_example_ids"] = list(example_ids)
        torch.save(payload, tensor_path)
        logger.info("Saved embeddings to %s", tensor_path)

        metadata = {
            "job_id": job.job_id,
            "model": job.model,
            "layer": job.layer,
            "dataset": job.dataset,
            "count": total_processed,
            "device": self._device,
            "chat_messages_field": job.chat_messages_field,
            "chat_template_used": chat_template_used,
            "paired_chat_messages_field": job.paired_chat_messages_field,
            "paired_text_field": job.paired_text_field,
            "paired_chat_template_used": paired_chat_template_used,
            "example_ids": list(example_ids),
        }
        if paired_embeddings:
            metadata["paired_count"] = len(paired_embeddings)
            metadata["paired_example_ids"] = list(example_ids)
        self._storage.write_metadata(metadata_path, metadata)

    def load_embeddings(self, job_id: str) -> Dict[str, torch.Tensor]:
        tensor_path = self._storage.embedding_tensor_path(job_id)
        return torch.load(tensor_path, map_location="cpu")

    def load_metadata(self, job_id: str) -> Dict:
        metadata_path = self._storage.embedding_metadata_path(job_id)
        return self._storage.read_metadata(metadata_path)

    def _build_input_text(
        self,
        record: Dict[str, Any],
        job: EmbeddingJobConfig,
        tokenizer: AutoTokenizer,
        *,
        chat_field: Optional[str] = None,
        text_field: Optional[str] = None,
    ) -> tuple[str, bool]:
        effective_chat_field = chat_field if chat_field is not None else job.chat_messages_field
        effective_text_field = text_field if text_field is not None else job.text_field
        if effective_chat_field:
            raw_messages = record.get(effective_chat_field)
            messages = self._normalize_messages(raw_messages)
            if not messages:
                raise ValueError(f"Record missing chat messages field '{effective_chat_field}'")
            if hasattr(tokenizer, "apply_chat_template"):
                template_kwargs = job.chat_template_kwargs or {}
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=job.chat_add_generation_prompt,
                    **template_kwargs,
                )
                return text, True
            logger.warning(
                "Tokenizer %s does not support chat templates; falling back to ad-hoc join",
                tokenizer.__class__.__name__,
            )
            return self._join_messages(messages), False

        field = effective_text_field or job.prompt_field
        if field is None:
            raise ValueError("Embedding job must define either chat_messages_field or text/prompt field")
        if field not in record:
            raise KeyError(f"Record missing field '{field}' for embedding job {job.job_id}")
        payload = record[field]
        if isinstance(payload, (list, tuple)):
            payload = "\n".join(str(item) for item in payload)
        if not isinstance(payload, str):
            payload = json.dumps(payload, ensure_ascii=False)
        return payload, False

    @staticmethod
    def _normalize_messages(raw: Any) -> Sequence[Dict[str, str]]:
        if raw is None:
            return []
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return [{"role": "user", "content": raw}]
            return EmbeddingCacheManager._normalize_messages(parsed)
        if isinstance(raw, dict):
            if "messages" in raw:
                return EmbeddingCacheManager._normalize_messages(raw["messages"])
            if "role" in raw and "content" in raw:
                return [{"role": str(raw.get("role", "user")), "content": str(raw.get("content", ""))}]
            # fallback: treat values as concatenated string
            content = json.dumps(raw, ensure_ascii=False)
            return [{"role": "user", "content": content}]
        if isinstance(raw, Iterable):
            normalized: List[Dict[str, str]] = []
            for item in raw:
                if isinstance(item, dict):
                    role = str(item.get("role", "user"))
                    content = item.get("content")
                    if content is None and "text" in item:
                        content = item["text"]
                    if isinstance(content, (list, tuple)):
                        content = "\n".join(str(x) for x in content)
                    if content is None:
                        content = json.dumps(item, ensure_ascii=False)
                    normalized.append({"role": role, "content": str(content)})
                elif isinstance(item, str):
                    normalized.append({"role": "user", "content": item})
                else:
                    normalized.append({"role": "user", "content": json.dumps(item, ensure_ascii=False)})
            return normalized
        return []

    @staticmethod
    def _join_messages(messages: Sequence[Dict[str, str]]) -> str:
        lines = [f"[{msg.get('role', 'user')}]: {msg.get('content', '')}" for msg in messages]
        return "\n".join(lines)
