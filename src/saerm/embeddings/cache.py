from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable, List, Optional

import torch
from datasets import Dataset, IterableDataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from ..config import EmbeddingJobConfig
from ..storage import StorageManager
from ..verification import verify_embeddings
from .rendering import ensure_chat_messages, render_messages

if "TRANSFORMERS_CACHE" in os.environ and "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = os.environ["TRANSFORMERS_CACHE"]

logger = logging.getLogger(__name__)

class EmbeddingCacheManager:
    """Create and persist embeddings for downstream SAE training."""

    def __init__(self, storage: StorageManager, device: Optional[str] = None) -> None:
        self._storage = storage
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def run_job(self, job: EmbeddingJobConfig, dataset: Dataset | IterableDataset) -> None:
        logger.info("Running embedding job %s", job.job_id)

        use_bf16 = isinstance(self._device, str) and self._device.startswith("cuda") and torch.cuda.is_available()
        model_kwargs: Dict[str, Any] = {"output_hidden_states": True}
        if use_bf16:
            model_kwargs["torch_dtype"] = torch.bfloat16
        model = AutoModel.from_pretrained(job.model, **model_kwargs)
        model.to(self._device)
        model.eval()

        tokenizer_name = job.tokenizer or job.model
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        prompt_field = job.prompt_field or "prompt"
        chosen_field = job.chosen_field or "chosen"
        rejected_field = job.rejected_field or "rejected"

        embeddings: List[torch.Tensor] = []
        records: List[Dict[str, Any]] = []

        batch_size = max(1, int(getattr(job, "batch_size", 16) or 16))
        batch_texts: List[str] = []
        batch_records: List[Dict[str, Any]] = []

        iterable: Iterable = dataset

        for idx, record in enumerate(tqdm(iterable, desc=f"embeddings:{job.job_id}")):
            if job.max_examples is not None and idx >= job.max_examples:
                break

            prompt_value = record.get(prompt_field)
            for choice, field_name in (("chosen", chosen_field), ("rejected", rejected_field)):
                raw_value = record[field_name]
                # If list of plain strings, expand to multiple candidates; if chat messages or single, keep as one
                candidates: List[str] = []
                if isinstance(raw_value, list) and (len(raw_value) == 0 or not (isinstance(raw_value[0], dict) and "role" in raw_value[0])):
                    for item in raw_value:
                        normalized_messages = ensure_chat_messages(item, prompt_value)
                        rendered = render_messages(normalized_messages, tokenizer)
                        candidates.append(rendered)
                else:
                    normalized_messages = ensure_chat_messages(raw_value, prompt_value)
                    rendered = render_messages(normalized_messages, tokenizer)
                    candidates.append(rendered)

                for rendered in candidates:
                    batch_texts.append(rendered)
                    batch_records.append({
                        "dataset": job.dataset,
                        "example_index": idx,
                        "choice": choice,
                        "text": rendered,
                    })

                if len(batch_texts) >= batch_size:
                    batch_embeddings = self._encode_text_batch(model, tokenizer, batch_texts)
                    embeddings.append(batch_embeddings)
                    records.extend(batch_records)
                    batch_texts.clear()
                    batch_records.clear()

        # flush any remainder
        if batch_texts:
            batch_embeddings = self._encode_text_batch(model, tokenizer, batch_texts)
            embeddings.append(batch_embeddings)
            records.extend(batch_records)
            batch_texts.clear()
            batch_records.clear()

        if not embeddings:
            raise ValueError(f"No embeddings generated for job {job.job_id}")

        embedding_tensor = torch.cat(embeddings, dim=0)
        if embedding_tensor.ndim != 2:
            raise ValueError(f"Expected 2D embeddings tensor, found shape {tuple(embedding_tensor.shape)}")

        tensor_path = self._storage.embedding_tensor_path(job.job_id)
        payload: Dict[str, Any] = {
            "embeddings": embedding_tensor,
            "records": records,
        }
        torch.save(payload, tensor_path)
        logger.info(
            "Saved %d embeddings (dimension %d) for job %s to %s",
            embedding_tensor.shape[0],
            embedding_tensor.shape[1],
            job.job_id,
            tensor_path,
        )

        verification = verify_embeddings(
            payload,
            expected_dim=embedding_tensor.shape[1],
            min_std=1e-3,
            min_unique=min(64, embedding_tensor.shape[0]),
            require_records=True,
            expected_choices=("chosen", "rejected"),
        )
        if verification["ok"]:
            logger.info(
                "Embedding verification passed for job %s | examples=%d dim=%d mean_std=%.3e unique=%d",
                job.job_id,
                verification["num_examples"],
                verification["embedding_dim"],
                verification["mean_std"],
                verification["unique_examples"],
            )
        else:
            logger.warning(
                "Embedding verification issues for job %s: %s",
                job.job_id,
                "; ".join(verification["issues"]),
            )

        metadata_path = self._storage.embedding_metadata_path(job.job_id)
        metadata = {
            "job_id": job.job_id,
            "model": job.model,
            # 'layer' removed from config; we always use last layer
            "dataset": job.dataset,
            "examples": len(records) // 2,
            "embedding_dim": embedding_tensor.shape[1],
            "device": str(self._device),
            "max_examples": job.max_examples,
            "verification": {
                "num_examples": verification["num_examples"],
                "embedding_dim": verification["embedding_dim"],
                "mean_std": verification["mean_std"],
                "min_std": verification["min_std"],
                "unique_examples": verification["unique_examples"],
                "issues": verification["issues"],
            },
        }
        self._storage.write_metadata(metadata_path, metadata)

    def load_embeddings(self, job_id: str) -> Dict[str, Any]:
        tensor_path = self._storage.embedding_tensor_path(job_id)
        return torch.load(tensor_path, map_location="cpu")

    def load_metadata(self, job_id: str) -> Dict[str, Any]:
        metadata_path = self._storage.embedding_metadata_path(job_id)
        return self._storage.read_metadata(metadata_path)

    def _encode_text_batch(
        self,
        model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        texts: List[str],
    ) -> torch.Tensor:
        tokens = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        tokens = {name: tensor.to(self._device) for name, tensor in tokens.items()}
        use_cuda = isinstance(self._device, str) and self._device.startswith("cuda") and torch.cuda.is_available()
        with torch.inference_mode():
            if use_cuda:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = model(**tokens, output_hidden_states=True)
            else:
                outputs = model(**tokens, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden states; set output_hidden_states=True")

        # Always take last layer
        layer_tensor = hidden_states[-1]

        # If per-token, select last non-padding/EOS token per sequence; else it's already [batch, hidden]
        if layer_tensor.ndim == 3:
            input_ids = tokens.get("input_ids")
            attention_mask = tokens.get("attention_mask")
            if input_ids is None:
                raise RuntimeError("Tokenizer did not return input_ids; cannot select last token")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

            eos_token_id = getattr(tokenizer, "eos_token_id", None)
            if eos_token_id is None:
                is_not_eos = torch.ones_like(input_ids, dtype=torch.bool)
            elif isinstance(eos_token_id, (list, tuple, set)):
                is_not_eos = torch.ones_like(input_ids, dtype=torch.bool)
                for _id in eos_token_id:
                    is_not_eos &= (input_ids != int(_id))
            else:
                is_not_eos = (input_ids != int(eos_token_id))

            valid_mask = attention_mask.bool() & is_not_eos

            seq_len = input_ids.shape[1]
            arange_idx = torch.arange(seq_len, device=input_ids.device).view(1, -1)

            last_valid_idx = (valid_mask * arange_idx).argmax(dim=1)
            last_nonpad_idx = (attention_mask.bool() * arange_idx).argmax(dim=1)
            any_valid = valid_mask.any(dim=1)
            last_idx = torch.where(any_valid, last_valid_idx, last_nonpad_idx)

            gathered = layer_tensor[torch.arange(layer_tensor.shape[0], device=layer_tensor.device), last_idx]
            cls_embeddings = gathered
        else:
            cls_embeddings = layer_tensor

        return cls_embeddings.cpu()
