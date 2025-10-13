from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from ..mounted import MountedSAE


@dataclass
class AutointerpLabel:
    dim: int
    text: str
    score: float
    label: str


def _ensure_initialized(mounted: MountedSAE, encode_fn: Callable[[Sequence[str]], object], device_str: Optional[str], sample_text: str) -> None:
    """Ensure `mounted.sae` is initialized by running a tiny forward pass."""
    with torch.no_grad():
        batch = encode_fn([sample_text])
        _ = mounted.forward(batch)  # populates cached activation and initializes SAE lazily


def get_top_text_per_dimension(
    mounted: MountedSAE,
    texts: Sequence[str],
    encode_fn: Callable[[Sequence[str]], object],
    *,
    batch_size: int = 64,
    show_progress: bool = False,
) -> Tuple[List[str], List[float]]:
    """Return the single most-activating text and score for each SAE dimension.

    Args:
        mounted: A ready `MountedSAE` instance (with base model and SAE weights loaded).
        texts: Corpus of strings to probe.
        encode_fn: Callable that turns a list of strings into model inputs (e.g., tokenizer(...)).
        batch_size: Number of texts to process per batch.
        show_progress: If True, shows a tqdm progress bar.

    Returns:
        (top_texts, top_scores), each of length equal to the number of SAE neurons.
    """
    if len(texts) == 0:
        return [], []

    # Initialize SAE and infer dimensionality
    _ensure_initialized(mounted, encode_fn, getattr(mounted, "base_device", None), texts[0])

    # Infer number of neurons from a tiny forward pass
    with torch.no_grad():
        batch0 = encode_fn(texts[: min(2, len(texts))])
        _xhat, info0 = mounted.forward(batch0)
        codes0: torch.Tensor = info0["codes"]  # shape: [B, M]
        num_neurons = int(codes0.shape[1])

    # Track global maxima per dimension
    top_scores = torch.full((num_neurons,), float("-inf"))
    top_indices = torch.full((num_neurons,), -1, dtype=torch.long)

    # Iterate corpus
    total = len(texts)
    rng = range(0, total, batch_size)
    if show_progress:
        try:
            from tqdm.auto import tqdm  # type: ignore

            rng = tqdm(rng, desc="Autointerp: scanning texts", leave=False)  # type: ignore[assignment]
        except Exception:
            pass

    with torch.no_grad():
        for start in rng:  # type: ignore[arg-type]
            end = min(total, start + batch_size)
            batch_texts = texts[start:end]
            inputs = encode_fn(batch_texts)
            _xhat, info = mounted.forward(inputs)
            codes: torch.Tensor = info["codes"].detach()  # [B, M]
            # Per-dimension maxima within this batch
            batch_max_vals, batch_argmax = codes.max(dim=0)  # [M], [M] indices 0..B-1

            # Update global maxima where improved
            better_mask = batch_max_vals > top_scores
            if better_mask.any():
                better_idx = better_mask.nonzero(as_tuple=False).squeeze(1)
                top_scores[better_idx] = batch_max_vals[better_idx]
                # Global index of the text: start + argmax_in_batch
                top_indices[better_idx] = batch_argmax[better_idx] + start

    # Materialize top texts and scores
    top_texts: List[str] = []
    top_vals: List[float] = []
    for j in range(num_neurons):
        idx = int(top_indices[j].item())
        if idx >= 0 and idx < len(texts):
            top_texts.append(texts[idx])
            top_vals.append(float(top_scores[j].item()))
        else:
            top_texts.append("")
            top_vals.append(float("-inf"))

    return top_texts, top_vals


def label_concepts_from_top_texts(
    mounted: MountedSAE,
    texts: Sequence[str],
    encode_fn: Callable[[Sequence[str]], object],
    llm_complete: Callable[[str], str],
    *,
    prompt_template: Optional[str] = None,
    batch_size: int = 64,
    show_progress: bool = False,
) -> Dict[int, AutointerpLabel]:
    """Label each SAE dimension by prompting an external language model with its top-activating text.

    Args:
        mounted: MountedSAE instance.
        texts: Corpus to search for top activations.
        encode_fn: Text -> model inputs callable.
        llm_complete: Callable(prompt) -> string label; implemented elsewhere.
        prompt_template: Optional custom prompt. Must include "{text}" placeholder.
        batch_size: Batch size to scan texts.
        show_progress: If True, show progress while scanning texts.

    Returns:
        Mapping from SAE dimension index to `AutointerpLabel` containing the chosen top text, score, and label.
    """
    default_template = (
        "You are labeling a sparse autoencoder concept from a language model.\n"
        "Given the following text that strongly activates an SAE feature, provide a concise label (3-6 words)\n"
        "describing the underlying concept captured by this feature. Do not include punctuation.\n\n"
        "Text:\n{text}\n\n"
        "Label:"
    )
    tmpl = prompt_template or default_template

    top_texts, top_scores = get_top_text_per_dimension(
        mounted, texts, encode_fn, batch_size=batch_size, show_progress=show_progress
    )

    results: Dict[int, AutointerpLabel] = {}
    for dim_idx, (text, score) in enumerate(zip(top_texts, top_scores)):
        if text.strip() == "":
            label = ""
        else:
            prompt = tmpl.format(text=text)
            try:
                label = str(llm_complete(prompt)).strip()
            except Exception:
                label = ""
        results[int(dim_idx)] = AutointerpLabel(dim=int(dim_idx), text=text, score=float(score), label=label)

    return results


