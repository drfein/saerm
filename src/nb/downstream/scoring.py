from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.nb.datasets.base import format_conversation
from src.nb.nullbias.probe import get_rewards_both, get_rewards_with_nulling


def _dtype_from_name(name: str) -> torch.dtype:
    lowered = str(name).lower()
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp16", "float16", "half"}:
        return torch.float16
    if lowered in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype name: {name}")


def load_probe_tensor(path: str | Path | None) -> torch.Tensor | None:
    """Load a saved probe tensor from a torch or JSON artifact."""
    if path is None:
        return None

    probe_path = Path(path)
    if not probe_path.exists():
        raise FileNotFoundError(f"Probe file not found: {probe_path}")

    if probe_path.suffix == ".json":
        with probe_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            if "probe" in data:
                data = data["probe"]
            elif "probe_direction" in data:
                data = data["probe_direction"]
        probe = torch.tensor(data, dtype=torch.float32)
    else:
        loaded = torch.load(probe_path, map_location="cpu")
        if isinstance(loaded, dict):
            if "probe" in loaded:
                loaded = loaded["probe"]
            elif "probe_direction" in loaded:
                loaded = loaded["probe_direction"]
        if not isinstance(loaded, torch.Tensor):
            loaded = torch.tensor(loaded, dtype=torch.float32)
        probe = loaded.float().cpu()

    if probe.dim() not in {1, 2}:
        raise ValueError(f"Expected 1D or 2D probe tensor, got shape={tuple(probe.shape)}")
    return probe


def _format_pairs(
    tokenizer: Any,
    prompts: Sequence[str],
    responses: Sequence[str],
    *,
    message_histories: Sequence[Sequence[dict[str, str]] | None] | None = None,
    force_pair: bool = False,
) -> list[str] | list[tuple[str, str]]:
    if len(prompts) != len(responses):
        raise ValueError("prompts and responses must have the same length")
    if message_histories is not None and len(message_histories) != len(prompts):
        raise ValueError("message_histories and prompts must have the same length")

    formatted: list[str] | list[tuple[str, str]] = []
    for idx, (prompt, response) in enumerate(zip(prompts, responses)):
        history = None if message_histories is None else message_histories[idx]
        if history:
            if force_pair or not (hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None):
                transcript_parts = []
                for msg in history:
                    role = str(msg.get("role", "user")).strip().capitalize()
                    content = str(msg.get("content", "")).strip()
                    transcript_parts.append(f"{role}: {content}")
                transcript = "\n\n".join(transcript_parts)
                formatted.append((transcript, response))
            else:
                conv = [
                    {"role": str(msg.get("role", "user")), "content": str(msg.get("content", ""))}
                    for msg in history
                ]
                conv.append({"role": "assistant", "content": response})
                rendered = tokenizer.apply_chat_template(
                    conv,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                if tokenizer.bos_token is not None and rendered.startswith(tokenizer.bos_token):
                    rendered = rendered[len(tokenizer.bos_token):]
                formatted.append(rendered)
        else:
            formatted.append(format_conversation(tokenizer, prompt=prompt, response=response, force_pair=force_pair))
    return formatted


class RewardModelScorer:
    """Scores prompt/response pairs with an optional null-space projection probe."""

    def __init__(
        self,
        *,
        model_path: str,
        probe_path: str | Path | None = None,
        alpha: float = 1.0,
        device: str = "cuda",
        batch_size: int = 8,
        max_length: int = 2048,
        trust_remote_code: bool = True,
        torch_dtype: str = "bfloat16",
        force_pair_format: bool = False,
        show_progress: bool = False,
    ) -> None:
        self.model_path = model_path
        self.alpha = float(alpha)
        self.device = device
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.trust_remote_code = bool(trust_remote_code)
        self.force_pair_format = bool(force_pair_format)
        self.torch_dtype = _dtype_from_name(torch_dtype)
        self.show_progress = bool(show_progress)
        self.probe = load_probe_tensor(probe_path)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=self.torch_dtype,
        ).to(self.device)
        self.model.eval()
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def _formatted_inputs(
        self,
        prompts: Sequence[str],
        responses: Sequence[str],
        *,
        message_histories: Sequence[Sequence[dict[str, str]] | None] | None = None,
    ) -> list[str] | list[tuple[str, str]]:
        return _format_pairs(
            self.tokenizer,
            prompts,
            responses,
            message_histories=message_histories,
            force_pair=self.force_pair_format,
        )

    def score_pairs(
        self,
        prompts: Sequence[str],
        responses: Sequence[str],
        *,
        message_histories: Sequence[Sequence[dict[str, str]] | None] | None = None,
        use_nulling: bool = False,
    ) -> list[float]:
        inputs = self._formatted_inputs(prompts, responses, message_histories=message_histories)
        scores = get_rewards_with_nulling(
            self.model,
            self.tokenizer,
            inputs,
            probe=self.probe if use_nulling else None,
            alpha=self.alpha,
            batch_size=self.batch_size,
            device=self.device,
            max_length=self.max_length,
            show_progress=self.show_progress,
        )
        return [float(x) for x in scores.tolist()]

    def score_pairs_both(
        self,
        prompts: Sequence[str],
        responses: Sequence[str],
        *,
        message_histories: Sequence[Sequence[dict[str, str]] | None] | None = None,
    ) -> tuple[list[float], list[float]]:
        inputs = self._formatted_inputs(prompts, responses, message_histories=message_histories)
        baseline, nulled = get_rewards_both(
            self.model,
            self.tokenizer,
            inputs,
            probe=self.probe,
            alpha=self.alpha,
            batch_size=self.batch_size,
            device=self.device,
            max_length=self.max_length,
            show_progress=self.show_progress,
        )
        return (
            [float(x) for x in baseline.tolist()],
            [float(x) for x in nulled.tolist()],
        )


@lru_cache(maxsize=4)
def get_cached_scorer(
    model_path: str,
    probe_path: str | None,
    alpha: float,
    device: str,
    batch_size: int,
    max_length: int,
    trust_remote_code: bool,
    torch_dtype: str,
    force_pair_format: bool,
    show_progress: bool,
) -> RewardModelScorer:
    return RewardModelScorer(
        model_path=model_path,
        probe_path=probe_path,
        alpha=alpha,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        force_pair_format=force_pair_format,
        show_progress=show_progress,
    )
