from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import math
import torch
from transformers import AutoTokenizer

from ..config import (
    AutoInterpJobConfig,
    DEFAULT_AUTOINTERP_ASSISTANT_PROMPT,
    EmbeddingJobConfig,
    ExperimentConfig,
    HeadTrainingConfig,
    SAETrainingConfig,
)
from ..data.datasets import DatasetLike, DatasetManager
from ..embeddings.rendering import ensure_chat_messages, render_messages
from ..heads.base import HeadFactory
from ..llm import GeminiLLMClient, generate_llm_response
from ..storage import StorageManager
from ..sae.inference import SAEFeatureExtractor

try:  # pragma: no cover - optional dependency
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

logger = logging.getLogger(__name__)

_EXPLANATION_PATTERN = re.compile(r"explanation\s*:\s*(.+)", re.IGNORECASE)
_SCORE_PATTERN = re.compile(r"score\s*:\s*([1-5])", re.IGNORECASE)


class AutoInterpreter:
    """Runs LLM-powered interpretations for SAE features."""

    def __init__(
        self,
        storage: StorageManager,
        job: AutoInterpJobConfig,
        *,
        config: Optional[ExperimentConfig] = None,
        dataset_manager: Optional[DatasetManager] = None,
        llm_client: Optional[GeminiLLMClient] = None,
    ) -> None:
        self._storage = storage
        self._job = job
        self._llm_client = llm_client
        self._config = config
        self._dataset_manager = dataset_manager
        if self._config is not None and self._dataset_manager is None:
            self._dataset_manager = DatasetManager(self._config)
        self._sae_job_map: Dict[str, SAETrainingConfig] = {}
        self._embedding_job_map: Dict[str, EmbeddingJobConfig] = {}
        self._head_job_map: Dict[str, HeadTrainingConfig] = {}
        if self._config is not None:
            self._sae_job_map = {cfg.job_id: cfg for cfg in self._config.sae_jobs}
            self._embedding_job_map = {cfg.job_id: cfg for cfg in self._config.embedding_jobs}
            self._head_job_map = {cfg.job_id: cfg for cfg in self._config.head_jobs}
        self._tokenizers: Dict[str, AutoTokenizer] = {}
        self._context_cache: Dict[Tuple[str, int], str] = {}
        self._dataset_resources: Optional[Tuple[EmbeddingJobConfig, DatasetLike, AutoTokenizer]] = None
        self._warned_dataset_access = False
        self._head_weights: Optional[np.ndarray] = None
        self._embedding_payload: Optional[Dict[str, Any]] = None
        self._sae_extractor: Optional[SAEFeatureExtractor] = None
        self._codes: Optional[torch.Tensor] = None
        self._normalized_embeddings: Optional[torch.Tensor] = None

    def run(self) -> Dict[str, Any]:
        examples_path = self._storage.sae_feature_examples_path(self._job.sae_job)
        if not examples_path.exists():
            raise FileNotFoundError(
                f"Feature examples not found for SAE job {self._job.sae_job} at {examples_path}"
            )

        with examples_path.open("r", encoding="utf-8") as handle:
            all_examples: Dict[str, List[Dict[str, Any]]] = json.load(handle)

        features = self._select_features(all_examples)
        features = self._apply_concept_limit(features)
        if not features:
            logger.warning(
                "No feature activations available for SAE job %s", self._job.sae_job
            )
            return {}

        client = self._llm_client or GeminiLLMClient(
            model_name=self._job.model_name,
            api_key=self._job.api_key,
        )
        assistant_prompt = (
            self._job.assistant_prompt or DEFAULT_AUTOINTERP_ASSISTANT_PROMPT
        )
        max_examples = max(0, self._job.max_examples_per_feature)
        if max_examples == 0:
            logger.warning(
                "AutoInterp job %s configured with zero examples per feature; nothing to do",
                self._job.job_id,
            )
            return {}

        self._hydrate_missing_contexts(all_examples, features, max_examples)

        feature_payloads: List[Tuple[str, str, List[Dict[str, Any]]]] = []
        for feature_id in features:
            items = all_examples.get(feature_id, [])
            if not items:
                continue
            selected = self._select_examples_balanced(feature_id, items, max_examples)
            prompt = self._build_prompt(feature_id, selected)
            feature_payloads.append((feature_id, prompt, selected))

        if not feature_payloads:
            logger.warning("No valid feature payloads generated for AutoInterp job %s", self._job.job_id)
            return {}

        progress: Optional[Any] = None
        if tqdm is not None:
            progress = tqdm(
                total=len(feature_payloads),
                desc=f"AutoInterp {self._job.job_id}",
                unit="feature",
            )

        batch_size = max(1, int(getattr(self._job, "batch_size", 1) or 1))
        output_path = self._storage.sae_autointerp_path(
            self._job.sae_job, self._job.job_id
        )
        summary_path = self._storage.sae_autointerp_summary_path(
            self._job.sae_job, self._job.job_id
        )
        model_name = getattr(client, "model_name", self._job.model_name)
        results: Dict[str, Dict[str, Any]] = {}

        for batch in self._batched(feature_payloads, batch_size):
            batch_results = self._process_batch(batch, client, assistant_prompt)
            results.update(batch_results)
            self._persist_results(
                output_path,
                results,
                assistant_prompt,
                max_examples,
                model_name,
            )
            self._persist_summary(
                summary_path,
                results,
            )
            if progress is not None:
                progress.update(len(batch))

        if progress is not None:
            progress.close()

        return {
            "job_id": self._job.job_id,
            "sae_job": self._job.sae_job,
            "model_name": model_name,
            "assistant_prompt": assistant_prompt,
            "max_examples_per_feature": max_examples,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "feature_results": results,
        }

    def _select_features(
        self, feature_map: Dict[str, List[Dict[str, Any]]]
    ) -> List[str]:
        requested_features: Optional[List[int]] = None
        if self._job.feature_ids:
            requested_features = [int(fid) for fid in self._job.feature_ids]
        elif self._job.head_job:
            requested_features = self._features_from_head()

        if not requested_features:
            return sorted(feature_map.keys(), key=self._feature_sort_key)

        selected: List[str] = []
        for fid in requested_features:
            key = str(fid)
            if key not in feature_map:
                logger.warning(
                    "Requested feature %s missing from SAE job %s",
                    key,
                    self._job.sae_job,
                )
                continue
            selected.append(key)
        return selected

    def _feature_sort_key(self, feature_id: str) -> Any:  # type: ignore[override]
        if feature_id.isdigit():
            return int(feature_id)
        return feature_id

    def _build_prompt(
        self, feature_id: str, examples: List[Dict[str, Any]]
    ) -> str:
        # Split examples into positives (activated) and negatives (non-activated but semantically similar)
        positives: List[Dict[str, Any]] = []
        negatives: List[Dict[str, Any]] = []
        for it in examples:
            try:
                act = float(it.get("activation", 0.0))
            except Exception:
                act = 0.0
            if act > 0.0:
                positives.append(it)
            else:
                negatives.append(it)

        header_lines = [
            f"You are interpreting feature {feature_id} in a language model.",
            "Positive examples are contexts where this feature had high activation.",
            "Negative examples are semantically similar contexts where this feature did not activate.",
            "Infer the concept present in the positive examples only. Do not include information that appears only in the negative examples.",
            "Respond with two lines:",
            "Explanation: <your concise concept explanation>",
            "Score: <an integer 1-5 for confidence>",
            "",
            "Positive (activated) examples:",
        ]

        pos_blocks: List[str] = []
        for idx, example in enumerate(positives, start=1):
            activation = self._format_number(example.get("activation"))
            context = example.get("text")
            if not context:
                context = self._hydrate_context(example)
            context = context or "[no context available]"
            pos_blocks.append("\n".join([f"Example {idx}:", f"Activation: {activation}", f"Context: {context}"]))

        neg_header = ["", "Negative (non-activated but similar) examples:"]
        neg_blocks: List[str] = []
        for idx, example in enumerate(negatives, start=1):
            activation = self._format_number(example.get("activation"))
            context = example.get("text")
            if not context:
                context = self._hydrate_context(example)
            context = context or "[no context available]"
            neg_blocks.append("\n".join([f"Example {idx}:", f"Activation: {activation}", f"Context: {context}"]))

        parts: List[str] = ["\n".join(header_lines)]
        if pos_blocks:
            parts.append("\n\n".join(pos_blocks))
        parts.append("\n".join(neg_header))
        if neg_blocks:
            parts.append("\n\n".join(neg_blocks))
        return "\n\n".join(parts)

    def _format_number(self, value: Any) -> str:
        if isinstance(value, (int, float)):
            return f"{value:.4f}"
        return str(value)

    def _parse_response(self, text: str) -> Dict[str, Any]:
        explanation_match = _EXPLANATION_PATTERN.search(text)
        score_match = _SCORE_PATTERN.search(text)

        explanation = explanation_match.group(1).strip() if explanation_match else ""
        score = int(score_match.group(1)) if score_match else None

        errors: List[str] = []
        if not explanation_match:
            errors.append("missing_explanation")
        if not score_match:
            errors.append("missing_score")

        return {"explanation": explanation, "score": score, "errors": errors or None}

    def _apply_concept_limit(self, features: List[str]) -> List[str]:
        limit = self._job.max_concepts
        if limit is None:
            return features
        if limit <= 0:
            return []
        return features[:limit]

    def _select_examples_balanced(self, feature_id: str, items: List[Dict[str, Any]], max_examples: int) -> List[Dict[str, Any]]:
        if max_examples <= 0:
            return []
        # Activated examples are the provided items list
        positives = items
        # Build non-activated examples by scanning cached embeddings and SAE codes
        try:
            feature_idx = int(feature_id)
        except Exception:
            feature_idx = -1

        negatives: List[Dict[str, Any]] = []
        if feature_idx >= 0:
            negatives = self._collect_nonactivated_examples(feature_idx, max_examples, positives)

        # Target equal split
        half = max_examples // 2
        selected_pos = positives[:min(len(positives), half)]
        selected_neg = negatives[:min(len(negatives), half)]
        selected = list(selected_pos) + list(selected_neg)
        remaining = max_examples - len(selected)
        if remaining > 0:
            # fill from the side with more available
            extra_pos = positives[len(selected_pos):]
            extra_neg = negatives[len(selected_neg):]
            i = j = 0
            while remaining > 0 and (i < len(extra_pos) or j < len(extra_neg)):
                if i < len(extra_pos):
                    selected.append(extra_pos[i]); i += 1; remaining -= 1
                    if remaining == 0:
                        break
                if j < len(extra_neg) and remaining > 0:
                    selected.append(extra_neg[j]); j += 1; remaining -= 1
        return selected

    def _collect_nonactivated_examples(
        self,
        feature_idx: int,
        max_needed: int,
        positives: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        payload = self._load_embedding_payload()
        extractor = self._get_sae_extractor()
        if payload is None or extractor is None:
            return []
        embeddings: Optional[torch.Tensor] = payload.get("embeddings")
        records = payload.get("records")
        if embeddings is None or records is None:
            return []
        codes = self._ensure_codes(extractor, embeddings)
        if codes is None:
            return []

        # avoid picking indices used in positives
        positive_indices: List[int] = []
        for it in positives:
            idx = it.get("dataset_index")
            if idx is None:
                idx = it.get("example_id")
            try:
                positive_indices.append(int(idx))
            except Exception:
                continue
        positive_set = set(positive_indices)

        with torch.no_grad():
            col = codes[:, feature_idx]
            non_activated_mask = (col <= 0)
            if not torch.any(non_activated_mask):
                return []
            normalized = self._get_normalized_embeddings(embeddings)
            if normalized is None:
                return []

            total_needed = max_needed
            chosen_neg_indices: List[int] = []
            seen = set()  # across all positives

            # Distribute target negatives roughly evenly among positives
            per_pos = max(1, math.ceil(total_needed / max(1, len(positive_indices))))
            for pos_idx in positive_indices:
                if len(chosen_neg_indices) >= total_needed:
                    break
                if pos_idx < 0 or pos_idx >= normalized.shape[0]:
                    continue
                query = normalized[pos_idx]
                sims = torch.matmul(normalized, query)
                # mask out positives and activated rows
                sims[~non_activated_mask] = -1e9
                if pos_idx < sims.shape[0]:
                    sims[pos_idx] = -1e9
                # take a small pool per positive
                k = min(per_pos * 4, sims.shape[0])
                top_vals, top_idx = torch.topk(sims, k=k, largest=True)
                for cand in top_idx.tolist():
                    if len(chosen_neg_indices) >= total_needed:
                        break
                    if cand in positive_set or cand in seen:
                        continue
                    seen.add(cand)
                    chosen_neg_indices.append(cand)

        neg_entries: List[Dict[str, Any]] = []
        for idx in chosen_neg_indices[:max_needed]:
            rec = records[idx]
            text = rec.get("text")
            choice = rec.get("choice")
            example_index = rec.get("example_index", idx)
            entry: Dict[str, Any] = {
                "example_id": int(example_index),
                "dataset_index": int(example_index),
                "activation": 0.0,
                "text": text,
                "source": choice,
            }
            neg_entries.append(entry)
        return neg_entries

    def _get_normalized_embeddings(self, embeddings: torch.Tensor) -> Optional[torch.Tensor]:
        if self._normalized_embeddings is not None:
            return self._normalized_embeddings
        try:
            vecs = embeddings
            norms = torch.norm(vecs, dim=1, keepdim=True)
            norms = torch.where(norms == 0, torch.ones_like(norms), norms)
            self._normalized_embeddings = (vecs / norms).cpu()
        except Exception:
            self._normalized_embeddings = None
        return self._normalized_embeddings

    def _load_embedding_payload(self) -> Optional[Dict[str, Any]]:
        if self._embedding_payload is not None:
            return self._embedding_payload
        if not self._sae_job_map or not self._embedding_job_map:
            return None
        embedding_job = self._resolve_embedding_job()
        if embedding_job is None:
            return None
        tensor_path = self._storage.embedding_tensor_path(embedding_job.job_id)
        try:
            payload = torch.load(tensor_path, map_location="cpu")
        except Exception:
            return None
        self._embedding_payload = payload
        return self._embedding_payload

    def _get_sae_extractor(self) -> Optional[SAEFeatureExtractor]:
        if self._sae_extractor is not None:
            return self._sae_extractor
        if not self._sae_job_map:
            return None
        sae_cfg = self._sae_job_map.get(self._job.sae_job)
        if sae_cfg is None:
            return None
        # read SAE metadata to infer dims
        meta_path = self._storage.sae_metadata_path(sae_cfg.job_id)
        try:
            meta = self._storage.read_metadata(meta_path)
        except Exception:
            return None
        checkpoint = self._storage.sae_checkpoint_path(sae_cfg.job_id)
        input_dim = meta.get("input_dim")
        hidden_dim = meta.get("hidden_size") or meta.get("num_neurons")
        k_active = meta.get("k_active")
        try:
            self._sae_extractor = SAEFeatureExtractor(
                checkpoint_path=str(checkpoint),
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                k_active=k_active,
                device="cpu",
            )
        except Exception:
            self._sae_extractor = None
        return self._sae_extractor

    def _ensure_codes(self, extractor: SAEFeatureExtractor, embeddings: torch.Tensor) -> Optional[torch.Tensor]:
        if self._codes is not None:
            return self._codes
        try:
            self._codes = extractor.transform(embeddings)
        except Exception:
            self._codes = None
        return self._codes

    def _hydrate_context(self, example: Dict[str, Any]) -> Optional[str]:
        if example.get("text"):
            return str(example["text"])
        resources = self._prepare_dataset_resources()
        if resources is None:
            return None

        dataset_index = self._resolve_dataset_index(example)
        if dataset_index is None:
            return None
        source = str(example.get("source") or "chosen")
        key = (source, dataset_index)
        cached = self._context_cache.get(key)
        if cached:
            example["text"] = cached
            return cached

        embedding_job, dataset, tokenizer = resources
        rendered = self._fetch_context_for_key(embedding_job, dataset, tokenizer, source, dataset_index)
        if rendered:
            self._context_cache[key] = rendered
            example["text"] = rendered
        return rendered

    def _resolve_dataset_index(self, example: Dict[str, Any]) -> Optional[int]:
        candidate = example.get("example_id")
        if candidate is None:
            candidate = example.get("dataset_index")
        if candidate is None:
            return None
        try:
            return int(candidate)
        except (TypeError, ValueError):
            return None

    def _hydrate_missing_contexts(
        self,
        feature_map: Dict[str, List[Dict[str, Any]]],
        features: List[str],
        max_examples: int,
    ) -> None:
        if max_examples <= 0:
            return
        resources = self._prepare_dataset_resources()
        if resources is None:
            return

        embedding_job, dataset, tokenizer = resources
        pending: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
        for feature_id in features:
            entries = feature_map.get(feature_id)
            if not entries:
                continue
            for entry in entries[:max_examples]:
                if entry.get("text"):
                    continue
                dataset_index = self._resolve_dataset_index(entry)
                if dataset_index is None:
                    continue
                source = str(entry.get("source") or "chosen")
                key = (source, dataset_index)
                cached = self._context_cache.get(key)
                if cached:
                    entry["text"] = cached
                    continue
                pending.setdefault(key, []).append(entry)

        if not pending:
            return

        for (source, dataset_index), entries in pending.items():
            rendered = self._fetch_context_for_key(embedding_job, dataset, tokenizer, source, dataset_index)
            if not rendered:
                continue
            self._context_cache[(source, dataset_index)] = rendered
            for entry in entries:
                entry["text"] = rendered

    def _prepare_dataset_resources(self) -> Optional[Tuple[EmbeddingJobConfig, DatasetLike, AutoTokenizer]]:
        if self._dataset_resources is not None:
            return self._dataset_resources
        if self._dataset_manager is None or not self._sae_job_map or not self._embedding_job_map:
            return None
        embedding_job = self._resolve_embedding_job()
        if embedding_job is None:
            return None
        dataset = self._dataset_manager.get(embedding_job.dataset)
        if not hasattr(dataset, "__getitem__"):
            if not self._warned_dataset_access:
                logger.warning(
                    "Dataset %s does not support random access; cannot hydrate context for AutoInterp job %s",
                    embedding_job.dataset,
                    self._job.job_id,
                )
                self._warned_dataset_access = True
            return None
        tokenizer = self._get_tokenizer(embedding_job)
        self._dataset_resources = (embedding_job, dataset, tokenizer)
        return self._dataset_resources

    def _fetch_context_for_key(
        self,
        embedding_job: EmbeddingJobConfig,
        dataset: DatasetLike,
        tokenizer: AutoTokenizer,
        source: str,
        dataset_index: int,
    ) -> Optional[str]:
        try:
            record = dataset[int(dataset_index)]
        except (IndexError, TypeError, ValueError):
            logger.warning(
                "Dataset index %s out of range for dataset %s in AutoInterp job %s",
                dataset_index,
                embedding_job.dataset,
                self._job.job_id,
            )
            return None

        field_name = self._field_name_for_source(embedding_job, source)
        if not field_name or field_name not in record:
            logger.warning(
                "Field %s missing in dataset %s for AutoInterp job %s",
                field_name,
                embedding_job.dataset,
                self._job.job_id,
            )
            return None
        prompt_field = embedding_job.prompt_field or "prompt"
        prompt_value = record.get(prompt_field)
        raw_messages = record[field_name]
        messages = ensure_chat_messages(raw_messages, prompt_value)
        return render_messages(messages, tokenizer)

    def _field_name_for_source(self, embedding_job: EmbeddingJobConfig, source: str) -> Optional[str]:
        normalized = "chosen" if source.lower() == "chosen" else "rejected"
        if normalized == "chosen":
            return embedding_job.chosen_field or "chosen"
        return embedding_job.rejected_field or "rejected"

    def _features_from_head(self) -> Optional[List[int]]:
        if not self._job.head_job:
            return None
        if not self._head_job_map:
            logger.warning("AutoInterp job %s requested head weights but config lacks head jobs", self._job.job_id)
            return None
        head_cfg = self._head_job_map.get(self._job.head_job)
        if head_cfg is None:
            logger.warning("Unknown head job %s referenced by AutoInterp job %s", self._job.head_job, self._job.job_id)
            return None
        if not head_cfg.sae_job:
            logger.warning("Head job %s does not reference an SAE job; cannot select features", head_cfg.job_id)
            return None
        if head_cfg.sae_job != self._job.sae_job:
            logger.warning(
                "Head job %s SAE %s does not match AutoInterp SAE %s",
                head_cfg.job_id,
                head_cfg.sae_job,
                self._job.sae_job,
            )
            return None

        model_path = self._storage.head_model_path(head_cfg.job_id)
        if not model_path.exists():
            logger.warning("Head model for %s not found at %s", head_cfg.job_id, model_path)
            return None
        try:
            head = HeadFactory.load(head_cfg.head_type, str(model_path))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to load head %s: %s", head_cfg.job_id, exc)
            return None

        weights = self._extract_head_weights(head)
        if weights is None or weights.size == 0:
            logger.warning("Head %s does not expose weights; cannot select features", head_cfg.job_id)
            return None

        top_n = max(1, int(self._job.head_top_n))
        flat = weights.flatten()
        self._head_weights = flat
        if flat.size <= top_n:
            indices = list(range(flat.size))
        else:
            pos_indices = np.argsort(flat)[-top_n:][::-1]
            neg_indices = np.argsort(flat)[:top_n]
            ordered = list(pos_indices) + [idx for idx in neg_indices if idx not in pos_indices]
            indices = ordered

        return [int(idx) for idx in indices]

    def _extract_head_weights(self, head) -> Optional[np.ndarray]:
        for attr in ("weights", "_weights"):
            value = getattr(head, attr, None)
            if value is not None:
                return np.asarray(value, dtype=float)
        model = getattr(head, "_model", None)
        if model is not None and hasattr(model, "coef_"):
            return np.asarray(model.coef_, dtype=float)
        return None

    def _persist_results(
        self,
        output_path: Path,
        results: Dict[str, Dict[str, Any]],
        assistant_prompt: str,
        max_examples: int,
        model_name: str,
    ) -> None:
        output = {
            "job_id": self._job.job_id,
            "sae_job": self._job.sae_job,
            "model_name": model_name,
            "assistant_prompt": assistant_prompt,
            "max_examples_per_feature": max_examples,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "feature_results": results,
        }
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2, sort_keys=True)
        logger.info(
            "Saved AutoInterp results for SAE job %s (%s features) to %s",
            self._job.sae_job,
            len(results),
            output_path,
        )

    def _persist_summary(
        self,
        output_path: Path,
        results: Dict[str, Dict[str, Any]],
    ) -> None:
        # Build a minimal map: feature_id -> {explanation, head_weight}
        summary: Dict[str, Dict[str, Any]] = {}
        for fid, item in results.items():
            summary[fid] = {
                "explanation": item.get("explanation"),
                "head_weight": item.get("head_weight"),
            }
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        logger.info(
            "Saved AutoInterp summary for SAE job %s (%s features) to %s",
            self._job.sae_job,
            len(summary),
            output_path,
        )

    def _process_batch(
        self,
        batch: List[Tuple[str, str, List[Dict[str, Any]]]],
        client: GeminiLLMClient,
        assistant_prompt: str,
    ) -> Dict[str, Dict[str, Any]]:
        if not batch:
            return {}
        desired_workers = max(1, int(getattr(self._job, "batch_size", len(batch)) or len(batch)))
        max_workers = min(desired_workers, len(batch))
        results: Dict[str, Dict[str, Any]] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(
                    generate_llm_response,
                    prompt,
                    assistant_prompt=assistant_prompt,
                    client=client,
                ): (feature_id, prompt, examples)
                for feature_id, prompt, examples in batch
            }

            for future in as_completed(future_map):
                feature_id, prompt, examples = future_map[future]
                try:
                    response_text = future.result()
                    parsed = self._parse_response(response_text)
                except Exception as exc:
                    logger.error(
                        "AutoInterp LLM request failed for feature %s: %s",
                        feature_id,
                        exc,
                    )
                    response_text = ""
                    parsed = {
                        "explanation": "",
                        "score": None,
                        "errors": ["generation_failed"],
                    }

                head_weight: Optional[float] = None
                if self._head_weights is not None:
                    try:
                        idx = int(feature_id)
                        if 0 <= idx < self._head_weights.size:
                            head_weight = float(self._head_weights[idx])
                    except Exception:
                        head_weight = None

                results[feature_id] = {
                    "response": response_text,
                    "explanation": parsed["explanation"],
                    "score": parsed["score"],
                    "parse_errors": parsed["errors"],
                    "examples": examples,
                    "head_weight": head_weight,
                }

        return results

    def _batched(
        self, items: List[Tuple[str, str, List[Dict[str, Any]]]], size: int
    ) -> Iterable[List[Tuple[str, str, List[Dict[str, Any]]]]]:
        for start in range(0, len(items), size):
            yield items[start : start + size]

    def _resolve_embedding_job(self) -> Optional[EmbeddingJobConfig]:
        if not self._sae_job_map:
            return None
        sae_cfg = self._sae_job_map.get(self._job.sae_job)
        if sae_cfg is None:
            logger.warning("Unknown SAE job %s referenced by AutoInterp job %s", self._job.sae_job, self._job.job_id)
            return None
        if not self._embedding_job_map:
            return None
        embedding_cfg = self._embedding_job_map.get(sae_cfg.embedding_job)
        if embedding_cfg is None:
            logger.warning(
                "SAE job %s references missing embedding job %s",
                sae_cfg.job_id,
                sae_cfg.embedding_job,
            )
        return embedding_cfg

    def _get_tokenizer(self, embedding_job: EmbeddingJobConfig) -> AutoTokenizer:
        key = embedding_job.tokenizer or embedding_job.model
        tokenizer = self._tokenizers.get(key)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(key)
            self._tokenizers[key] = tokenizer
        return tokenizer


__all__ = ["AutoInterpreter"]
