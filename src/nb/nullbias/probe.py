"""
Probe direction building from contrastive pairs.

Computes the bias direction using difference-of-means:
    probe = mean(positive_embeddings) - mean(negative_embeddings)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.nb.datasets.base import ContrastivePair

logger = logging.getLogger(__name__)

def _prepare_ortho_basis(
    basis: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Orthonormalize a probe basis once (shared across batches/alphas).

    Args:
        basis: [d] or [k, d]
        device: device to place the returned basis on

    Returns:
        [k', d] float32 orthonormal basis (k' may be 0)
    """
    if basis.dim() == 1:
        basis = basis.unsqueeze(0)
    if basis.numel() == 0 or basis.shape[0] == 0:
        return torch.zeros((0, basis.shape[-1]), device=device, dtype=torch.float32)
    ortho_basis = gram_schmidt([v for v in basis])
    if ortho_basis.shape[0] == 0:
        return torch.zeros((0, basis.shape[-1]), device=device, dtype=torch.float32)
    return ortho_basis.to(device=device, dtype=torch.float32)


def tokenize_inputs(
    tokenizer: AutoTokenizer,
    texts: List,
    padding: bool = True,
    truncation: bool = True,
    max_length: int = 2048,
    return_tensors: str = "pt",
) -> Dict[str, torch.Tensor]:
    """Tokenize inputs, handling both single strings and (prompt, response) pairs.
    
    Args:
        tokenizer: HuggingFace tokenizer
        texts: List of strings OR list of (prompt, response) tuples
        padding: Whether to pad
        truncation: Whether to truncate
        max_length: Maximum sequence length
        return_tensors: Return format
        
    Returns:
        Tokenized inputs dict
    """
    if not texts:
        raise ValueError("No texts provided")
    
    # Check if inputs are pairs (tuples) or single strings
    first = texts[0]
    if isinstance(first, tuple) and len(first) == 2:
        # Pair format: tokenizer(text_a, text_b)
        texts_a = [t[0] for t in texts]
        texts_b = [t[1] for t in texts]
        return tokenizer(
            texts_a,
            texts_b,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
        )
    else:
        # Single string format
        return tokenizer(
            texts,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
        )


def gram_schmidt(vectors: List[torch.Tensor]) -> torch.Tensor:
    """Orthogonalize vectors using Gram-Schmidt (returns an orthonormal basis).
    
    Args:
        vectors: List of 1D tensors [d]
        
    Returns:
        Orthonormal basis matrix [k, d] (k <= len(vectors))
    """
    if not vectors:
        raise ValueError("No vectors provided to gram_schmidt")
    
    basis: List[torch.Tensor] = []
    for v in vectors:
        v = v.float()
        for b in basis:
            v = v - (v @ b) * b
        norm = v.norm()
        if norm > 1e-8:
            basis.append(v / norm)
    
    # If all were near-zero, return empty basis with correct dim
    if not basis:
        return torch.zeros(0, vectors[0].shape[0])
    
    return torch.stack(basis, dim=0)


def project_to_null_space(hidden: torch.Tensor, basis: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Project hidden states to the null space of a basis (remove subspace components).
    
    Uses Gram-Schmidt to orthonormalize the basis, then projects out each direction.
    
    Args:
        hidden: [batch, d] hidden states to project
        basis: [d] single vector OR [k, d] multiple vectors (need not be orthonormal)
        alpha: Scaling factor for nulling (0=no nulling, 1=full nulling, >1=over-nulling)
        
    Returns:
        [batch, d] with the basis subspace removed (scaled by alpha).
    """
    if alpha == 0.0:
        return hidden
    
    # Handle 1D input: [d] -> [1, d]
    if basis.dim() == 1:
        basis = basis.unsqueeze(0)
    
    if basis.shape[0] == 0:
        return hidden
    
    # Orthonormalize using Gram-Schmidt (handles near-colinear vectors)
    ortho_basis = gram_schmidt([v for v in basis])  # [k', d] where k' <= k
    
    if ortho_basis.shape[0] == 0:
        return hidden
    
    # Project onto orthonormal span and remove (scaled by alpha)
    ortho_basis = ortho_basis.to(hidden.device).float()
    coeffs = hidden.float() @ ortho_basis.T  # [batch, k']
    projection = coeffs @ ortho_basis         # [batch, d]
    
    return hidden - (alpha * projection).to(hidden.dtype)


def get_base_model(model: AutoModelForSequenceClassification):
    """Extract base transformer from reward model wrapper.
    
    Args:
        model: Reward model (AutoModelForSequenceClassification)
        
    Returns:
        Base transformer model
        
    Raises:
        ValueError: If base model cannot be found
    """
    for attr in ["model", "transformer", "base_model"]:
        if hasattr(model, attr):
            return getattr(model, attr)
    raise ValueError(f"Cannot find base model in {type(model)}")


def get_score_head(model: AutoModelForSequenceClassification) -> torch.nn.Linear:
    """Extract the score/classification head from reward model.
    
    Args:
        model: Reward model
        
    Returns:
        Linear layer that maps hidden states to scores
        
    Raises:
        ValueError: If score head cannot be found
    """
    for attr in ["score", "classifier", "out_proj", "head", "reward_head", "score_head", "value_head", "v_head"]:
        if hasattr(model, attr):
            layer = getattr(model, attr)
            if isinstance(layer, torch.nn.Linear):
                return layer
    
    # Fallback: find any Linear with output_features=1
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) and module.out_features == 1:
            return module

    hidden_size = int(getattr(getattr(model, "config", None), "hidden_size", 0) or 0)
    named_parameters = dict(model.named_parameters())
    candidate_prefixes: List[Tuple[int, str]] = []
    for name, parameter in named_parameters.items():
        if not name.endswith(".weight") or parameter.ndim != 2 or parameter.shape[0] != 1:
            continue
        if hidden_size and parameter.shape[-1] != hidden_size:
            continue
        prefix = name[:-len(".weight")]
        priority = 1 if any(token in prefix.lower() for token in ("score", "class", "head", "reward", "value")) else 0
        candidate_prefixes.append((priority, prefix))

    if candidate_prefixes:
        _, prefix = max(candidate_prefixes)
        weight = named_parameters[f"{prefix}.weight"].detach()
        bias = named_parameters.get(f"{prefix}.bias")
        layer = torch.nn.Linear(weight.shape[-1], weight.shape[0], bias=bias is not None)
        layer = layer.to(device=weight.device, dtype=weight.dtype)
        layer.weight.data.copy_(weight)
        if bias is not None:
            layer.bias.data.copy_(bias.detach())
        return layer

    vector_candidate_prefixes: List[Tuple[int, str]] = []
    for name, parameter in named_parameters.items():
        if parameter.ndim != 1:
            continue
        is_weight_name = name.endswith(".weight")
        is_scalar_bias_name = name.endswith(".bias") and parameter.shape[0] == 1
        if not is_weight_name and not is_scalar_bias_name:
            continue
        prefix = name.rsplit(".", 1)[0]
        weight = named_parameters.get(f"{prefix}.weight")
        if weight is None and is_weight_name:
            weight = parameter
        if weight is None or weight.ndim != 1:
            continue
        if hidden_size and weight.shape[0] != hidden_size:
            continue
        priority = 1 if any(token in prefix.lower() for token in ("score", "class", "head", "reward", "value")) else 0
        vector_candidate_prefixes.append((priority, prefix))

    if vector_candidate_prefixes:
        _, prefix = max(vector_candidate_prefixes)
        weight = named_parameters[f"{prefix}.weight"].detach().view(1, -1)
        bias_param = named_parameters.get(f"{prefix}.bias")
        layer = torch.nn.Linear(weight.shape[-1], 1, bias=bias_param is not None)
        layer = layer.to(device=weight.device, dtype=weight.dtype)
        layer.weight.data.copy_(weight)
        if bias_param is not None:
            layer.bias.data.copy_(bias_param.detach().view(-1))
        return layer

    raise ValueError("Could not find score head in model")


def project_onto_null(u: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Project vector u onto the null space of basis vectors.
    
    Returns u with all components along basis vectors removed.
    
    Args:
        u: Vector to project [d]
        basis: Basis vectors to project out [k, d] or [d] for single vector
        
    Returns:
        Cleaned vector [d]
    """
    if basis.dim() == 1:
        basis = basis.unsqueeze(0)
    
    result = u.clone()
    for v in basis:
        v_norm_sq = torch.dot(v, v)
        if v_norm_sq > 1e-10:
            result = result - (torch.dot(result, v) / v_norm_sq) * v
    return result


def clean_probe(probe: torch.Tensor, nuisance_probes: List[torch.Tensor]) -> torch.Tensor:
    """Clean a probe by removing components along nuisance directions.
    
    Args:
        probe: The probe to clean [d]
        nuisance_probes: List of probes to project out
        
    Returns:
        Cleaned probe (renormalized) [d]
    """
    if not nuisance_probes:
        return probe
    basis = torch.stack([p.to(probe.device) for p in nuisance_probes], dim=0)
    cleaned = project_onto_null(probe, basis)
    return cleaned / (cleaned.norm() + 1e-8)


def clean_probe_with_correctness(
    *,
    probe: torch.Tensor,
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    correctness_pairs: List[ContrastivePair],
    batch_size: int = 8,
    device: str = "cuda",
    max_length: int = 2048,
) -> Tuple[torch.Tensor, Dict[str, Any], torch.Tensor]:
    """Clean a probe by projecting out a correctness direction learned from contrastive pairs.
    
    This is a common pattern across multiple experiments:
    1) Learn a "correct vs incorrect" probe direction
    2) Remove that direction from the target bias probe
    
    Args:
        probe: Target probe to clean [d]
        model: Reward model
        tokenizer: Tokenizer
        correctness_pairs: Contrastive pairs defining correctness direction
        batch_size: Batch size for embedding extraction
        device: Device to use
        max_length: Maximum sequence length
        
    Returns:
        Tuple of:
        - cleaned_probe: Cleaned probe (renormalized) [d]
        - metadata: Dict with correctness probe stats and overlap information
        - correctness_probe: The learned correctness direction [d]
    """
    if len(correctness_pairs) == 0:
        raise ValueError("No correctness pairs provided; cannot clean probe.")
    
    correctness_probe, corr_metadata = build_probe_direction(
        model=model,
        tokenizer=tokenizer,
        contrastive_pairs=correctness_pairs,
        batch_size=batch_size,
        device=device,
        max_length=max_length,
    )
    
    original_probe = probe.clone()
    cleaned_probe = clean_probe(probe, [correctness_probe])
    
    overlap = torch.dot(
        original_probe, correctness_probe.to(original_probe.device)
    ).abs().item()
    
    metadata = {
        "cleaned_with_correctness": True,
        "correctness_overlap": overlap,
        "correctness_probe_accuracy": corr_metadata.get("probe_accuracy", 0),
    }
    
    return cleaned_probe, metadata, correctness_probe


def clean_probe_basis_with_correctness(
    *,
    probes: List[torch.Tensor],
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    correctness_pairs: List[ContrastivePair],
    batch_size: int = 8,
    device: str = "cuda",
    max_length: int = 2048,
) -> Tuple[torch.Tensor, Dict[str, Any], torch.Tensor]:
    """Clean multiple probe directions by projecting out a correctness direction.
    
    Typical use: build several bias directions (e.g. position A-vs-rest, B-vs-rest, ...)
    and remove correctness, then orthonormalize and null the full subspace.
    
    Args:
        probes: List of raw probe directions [d]
        model/tokenizer/correctness_pairs: for learning correctness direction
        
    Returns:
        Tuple of:
        - basis: Orthonormal basis [k, d] after cleaning and Gram-Schmidt
        - metadata: cleaning metadata (includes per-vector overlaps)
        - correctness_probe: learned correctness direction [d]
    """
    if not probes:
        raise ValueError("No probe directions provided; cannot clean basis.")
    
    # Learn correctness direction once
    correctness_probe, corr_metadata = build_probe_direction(
        model=model,
        tokenizer=tokenizer,
        contrastive_pairs=correctness_pairs,
        batch_size=batch_size,
        device=device,
        max_length=max_length,
    )
    
    # Clean each probe direction against correctness
    overlaps: List[float] = []
    cleaned_list: List[torch.Tensor] = []
    for v in probes:
        v = v.to(correctness_probe.device)
        overlaps.append(float(torch.dot(v, correctness_probe.to(v.device)).abs().item()))
        cleaned_list.append(clean_probe(v, [correctness_probe]))
    
    # Orthonormalize the cleaned directions
    basis = gram_schmidt(cleaned_list)
    
    metadata: Dict[str, Any] = {
        "cleaned_with_correctness": True,
        "correctness_probe_accuracy": corr_metadata.get("probe_accuracy", 0),
        "correctness_overlap_mean": float(sum(overlaps) / max(len(overlaps), 1)),
        "correctness_overlap_per_direction": overlaps,
        "n_basis_vectors": int(basis.shape[0]),
    }
    
    return basis, metadata, correctness_probe


def get_embeddings(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: List[str],
    batch_size: int = 8,
    device: str = "cuda",
    max_length: int = 2048,
    show_progress: bool = True,
) -> torch.Tensor:
    """Extract last-token hidden state embeddings from reward model.
    
    Tokenizes on-the-fly per batch to avoid OOM on large datasets.
    
    Args:
        model: Reward model
        tokenizer: Tokenizer
        texts: List of formatted conversation texts
        batch_size: Batch size for inference
        device: Device to use
        max_length: Maximum sequence length
        show_progress: Whether to show progress bar
        
    Returns:
        [n_texts, hidden_dim] tensor of embeddings
    """
    model.eval()
    all_embeddings = []
    base_model = get_base_model(model)
    
    n_batches = (len(texts) + batch_size - 1) // batch_size
    
    logger.info("Processing %d texts in %d batches (tokenizing on-the-fly)...", len(texts), n_batches)
    
    iterator = range(n_batches)
    if show_progress:
        iterator = tqdm(iterator, desc="Extracting embeddings", total=n_batches)
    
    with torch.no_grad():
        for batch_idx in iterator:
            start = batch_idx * batch_size
            end = min(start + batch_size, len(texts))
            batch_texts = texts[start:end]
            
            # Tokenize just this batch (handles both strings and pairs)
            inputs = tokenize_inputs(
                tokenizer,
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = base_model(**inputs)
            if hasattr(outputs, "last_hidden_state"):
                hidden_states = outputs.last_hidden_state
            elif hasattr(outputs, "hidden_states"):
                hidden_states = outputs.hidden_states[-1]
            else:
                # Fallback for some models
                hidden_states = outputs[0]
            
            # Get last non-padding token for each example
            attention_mask = inputs["attention_mask"]
            last_token_indices = attention_mask.sum(dim=1) - 1
            batch_embeddings = hidden_states[
                torch.arange(hidden_states.size(0), device=device),
                last_token_indices,
            ]
            all_embeddings.append(batch_embeddings.float().cpu())
    
    return torch.cat(all_embeddings, dim=0)


def build_probe_direction(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    contrastive_pairs: List[ContrastivePair],
    batch_size: int = 8,
    device: str = "cuda",
    max_length: int = 2048,
    return_embeddings: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any]] | Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]:
    """Build probe direction using difference-of-means.
    
    The probe direction points from negative to positive:
        probe = mean(positive_embeddings) - mean(negative_embeddings)
    
    Args:
        model: Reward model
        tokenizer: Tokenizer
        contrastive_pairs: List of ContrastivePair objects
        batch_size: Batch size for embedding extraction
        device: Device to use
        max_length: Maximum sequence length
        
    Returns:
        Tuple of:
        - probe: [hidden_dim] normalized probe direction tensor
        - metadata: Dictionary with probe statistics
        - activations: Optional activation payload if return_embeddings=True
    """
    positive_texts = [pair.positive_text for pair in contrastive_pairs]
    negative_texts = [pair.negative_text for pair in contrastive_pairs]
    
    logger.info("Extracting embeddings for %d positive examples", len(positive_texts))
    positive_emb = get_embeddings(
        model, tokenizer, positive_texts, batch_size, device, max_length
    )
    
    logger.info("Extracting embeddings for %d negative examples", len(negative_texts))
    negative_emb = get_embeddings(
        model, tokenizer, negative_texts, batch_size, device, max_length
    )
    
    # Compute means
    positive_mean = positive_emb.mean(dim=0)
    negative_mean = negative_emb.mean(dim=0)
    
    # Difference of means
    probe = positive_mean - negative_mean
    raw_norm = probe.norm().item()
    probe = probe / (probe.norm() + 1e-8)
    
    # Compute statistics
    positive_proj = positive_emb @ probe
    negative_proj = negative_emb @ probe
    
    # Classification accuracy at optimal threshold
    threshold = (positive_proj.mean() + negative_proj.mean()) / 2
    correct = (positive_proj > threshold).sum() + (negative_proj <= threshold).sum()
    accuracy = float(correct) / (len(positive_proj) + len(negative_proj))
    
    metadata = {
        "n_positive": len(positive_texts),
        "n_negative": len(negative_texts),
        "hidden_dim": int(probe.shape[0]),
        "positive_mean_norm": float(positive_mean.norm()),
        "negative_mean_norm": float(negative_mean.norm()),
        "probe_raw_norm": raw_norm,
        "positive_proj_mean": float(positive_proj.mean()),
        "positive_proj_std": float(positive_proj.std()),
        "negative_proj_mean": float(negative_proj.mean()),
        "negative_proj_std": float(negative_proj.std()),
        "separation": float(positive_proj.mean() - negative_proj.mean()),
        "probe_accuracy": accuracy,
    }
    
    logger.info("Probe statistics:")
    logger.info("  Hidden dim: %d", metadata["hidden_dim"])
    logger.info("  Separation: %.4f", metadata["separation"])
    logger.info("  Probe accuracy: %.2f%%", 100 * metadata["probe_accuracy"])
    
    if return_embeddings:
        activations = {
            "positive_embeddings": positive_emb,
            "negative_embeddings": negative_emb,
            "pair_metadata": [pair.metadata for pair in contrastive_pairs],
        }
        return probe, metadata, activations

    return probe, metadata


def get_rewards_with_nulling(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: List[str],
    probe: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
    batch_size: int = 8,
    device: str = "cuda",
    max_length: int = 2048,
    show_progress: bool = True,
) -> torch.Tensor:
    """Compute reward scores with optional null-space projection.
    
    If probe is provided, projects hidden states onto the null space of the
    probe before computing scores.
    Tokenizes on-the-fly per batch to avoid OOM on large datasets.
    
    Args:
        model: Reward model
        tokenizer: Tokenizer  
        texts: List of formatted conversation texts
        probe: Optional probe direction tensor [hidden_dim] or [k, hidden_dim]
        alpha: Scaling factor for nulling (0=no nulling, 1=full nulling, >1=over-nulling)
        batch_size: Batch size for inference
        device: Device to use
        max_length: Maximum sequence length
        show_progress: Whether to show progress bar
        
    Returns:
        [n_texts] tensor of reward scores
    """
    model.eval()
    base_model = get_base_model(model)
    score_head = get_score_head(model)
    
    # Prepare probe for nulling
    do_nulling = probe is not None
    
    all_scores = []
    n_batches = (len(texts) + batch_size - 1) // batch_size
    
    logger.info("Processing %d texts in %d batches (tokenizing on-the-fly)...", len(texts), n_batches)
    
    iterator = range(n_batches)
    if show_progress:
        iterator = tqdm(iterator, desc="Computing rewards", total=n_batches)
    
    with torch.no_grad():
        for batch_idx in iterator:
            start = batch_idx * batch_size
            end = min(start + batch_size, len(texts))
            batch_texts = texts[start:end]
            
            # Tokenize just this batch (handles both strings and pairs)
            inputs = tokenize_inputs(
                tokenizer,
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            if do_nulling:
                # Get hidden states and null out probe direction
                outputs = base_model(**inputs)
                if hasattr(outputs, "last_hidden_state"):
                    hidden_states = outputs.last_hidden_state
                elif hasattr(outputs, "hidden_states"):
                    hidden_states = outputs.hidden_states[-1]
                else:
                    hidden_states = outputs[0]
                
                # Get last token hidden states
                attention_mask = inputs["attention_mask"]
                last_token_indices = attention_mask.sum(dim=1) - 1
                last_hidden = hidden_states[
                    torch.arange(hidden_states.size(0), device=device),
                    last_token_indices,
                ].float()
                
                # Null out probe subspace with alpha scaling
                nulled_hidden = project_to_null_space(last_hidden, probe, alpha=alpha)
                
                # Get score from nulled hidden state
                scores = score_head(nulled_hidden.to(hidden_states.dtype)).squeeze(-1)
            else:
                # Standard forward pass
                outputs = model(**inputs)
                scores = outputs.logits.squeeze(-1)
            
            all_scores.append(scores.cpu())
    
    return torch.cat(all_scores, dim=0)


def get_rewards_both(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: List[str],
    probe: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
    batch_size: int = 8,
    device: str = "cuda",
    max_length: int = 2048,
    show_progress: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute BOTH baseline and nulled rewards in a single forward pass.
    
    This is 2x faster than calling get_rewards_with_nulling twice.
    Tokenizes on-the-fly per batch to avoid OOM on large datasets.
    
    Args:
        model: Reward model
        tokenizer: Tokenizer  
        texts: List of formatted conversation texts
        probe: Probe direction tensor [hidden_dim] or [k, hidden_dim]
        alpha: Scaling factor for nulling (0=no nulling, 1=full nulling, >1=over-nulling)
        batch_size: Batch size for inference
        device: Device to use
        max_length: Maximum sequence length
        show_progress: Whether to show progress bar
        
    Returns:
        Tuple of (baseline_rewards, nulled_rewards) tensors, each [n_texts]
    """
    model.eval()
    base_model = get_base_model(model)
    score_head = get_score_head(model)
    
    # Prepare probe for nulling
    do_nulling = probe is not None
    
    all_baseline_scores = []
    all_nulled_scores = []
    n_batches = (len(texts) + batch_size - 1) // batch_size
    
    logger.info("Processing %d texts in %d batches (tokenizing on-the-fly)...", len(texts), n_batches)
    
    iterator = range(n_batches)
    if show_progress:
        iterator = tqdm(iterator, desc="Computing rewards (baseline+nulled)", total=n_batches)
    
    with torch.no_grad():
        for batch_idx in iterator:
            start = batch_idx * batch_size
            end = min(start + batch_size, len(texts))
            batch_texts = texts[start:end]
            
            # Tokenize just this batch (handles both strings and pairs)
            inputs = tokenize_inputs(
                tokenizer,
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Always get hidden states so we can compute both scores
            outputs = base_model(**inputs)
            if hasattr(outputs, "last_hidden_state"):
                hidden_states = outputs.last_hidden_state
            elif hasattr(outputs, "hidden_states"):
                hidden_states = outputs.hidden_states[-1]
            else:
                hidden_states = outputs[0]
            
            # Get last token hidden states
            attention_mask = inputs["attention_mask"]
            last_token_indices = attention_mask.sum(dim=1) - 1
            last_hidden = hidden_states[
                torch.arange(hidden_states.size(0), device=device),
                last_token_indices,
            ].float()
            
            # Baseline score (no nulling)
            baseline_scores = score_head(last_hidden.to(hidden_states.dtype)).squeeze(-1)
            all_baseline_scores.append(baseline_scores.cpu())
            
            # Nulled score with alpha scaling
            if do_nulling:
                nulled_hidden = project_to_null_space(last_hidden, probe, alpha=alpha)
                nulled_scores = score_head(nulled_hidden.to(hidden_states.dtype)).squeeze(-1)
            else:
                nulled_scores = baseline_scores
            all_nulled_scores.append(nulled_scores.cpu())
    
    return torch.cat(all_baseline_scores, dim=0), torch.cat(all_nulled_scores, dim=0)


def get_rewards_multi_alpha(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: List[str],
    probe: Optional[torch.Tensor],
    alphas: List[float],
    batch_size: int = 8,
    device: str = "cuda",
    max_length: int = 2048,
    show_progress: bool = True,
) -> Tuple[torch.Tensor, Dict[float, torch.Tensor]]:
    """Compute baseline rewards once and nulled rewards for many alphas efficiently.

    This avoids re-running the model forward pass for each alpha by:
      - running the base model once per batch to get last-token hidden states
      - computing the projection onto the probe subspace once per batch
      - scoring nulled hidden states for each alpha via: h - alpha * projection

    Args:
        model/tokenizer/texts: standard reward scoring inputs
        probe: probe direction [d] or basis [k, d]; if None, nulled==baseline for all alphas
        alphas: list of nulling strengths to evaluate

    Returns:
        baseline_scores: [n_texts]
        nulled_scores_by_alpha: dict mapping alpha -> [n_texts]
    """
    model.eval()
    base_model = get_base_model(model)
    score_head = get_score_head(model)

    if not alphas:
        raise ValueError("alphas must be non-empty")

    alphas_unique = list(dict.fromkeys(alphas))  # preserve order, drop dupes
    alphas_tensor = torch.tensor(alphas_unique, dtype=torch.float32, device=device)

    do_nulling = probe is not None
    ortho_basis = None
    if do_nulling:
        ortho_basis = _prepare_ortho_basis(probe, device=torch.device(device))

    all_baseline_scores: List[torch.Tensor] = []
    all_nulled_scores: Dict[float, List[torch.Tensor]] = {a: [] for a in alphas_unique}

    n_batches = (len(texts) + batch_size - 1) // batch_size
    logger.info(
        "Processing %d texts in %d batches (tokenizing on-the-fly)...",
        len(texts),
        n_batches,
    )

    iterator = range(n_batches)
    if show_progress:
        iterator = tqdm(iterator, desc="Computing rewards (baseline+multi-alpha)", total=n_batches)

    with torch.no_grad():
        for batch_idx in iterator:
            start = batch_idx * batch_size
            end = min(start + batch_size, len(texts))
            batch_texts = texts[start:end]

            inputs = tokenize_inputs(
                tokenizer,
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = base_model(**inputs)
            if hasattr(outputs, "last_hidden_state"):
                hidden_states = outputs.last_hidden_state
            elif hasattr(outputs, "hidden_states"):
                hidden_states = outputs.hidden_states[-1]
            else:
                hidden_states = outputs[0]

            attention_mask = inputs["attention_mask"]
            last_token_indices = attention_mask.sum(dim=1) - 1
            last_hidden = hidden_states[
                torch.arange(hidden_states.size(0), device=device),
                last_token_indices,
            ].float()  # [b, d] fp32 for stable projection

            baseline_scores = score_head(last_hidden.to(hidden_states.dtype)).squeeze(-1)
            all_baseline_scores.append(baseline_scores.cpu())

            if not do_nulling or ortho_basis is None or ortho_basis.shape[0] == 0:
                # No nulling; reuse baseline for each alpha
                for a in alphas_unique:
                    all_nulled_scores[a].append(baseline_scores.cpu())
                continue

            # Shared projection onto probe subspace
            coeffs = last_hidden @ ortho_basis.T  # [b, k]
            projection = coeffs @ ortho_basis     # [b, d]

            # Score each alpha without re-forwarding the model
            for a in alphas_unique:
                nulled_hidden = last_hidden - (float(a) * projection)
                nulled_scores = score_head(nulled_hidden.to(hidden_states.dtype)).squeeze(-1)
                all_nulled_scores[a].append(nulled_scores.cpu())

    baseline_out = torch.cat(all_baseline_scores, dim=0)
    nulled_out = {a: torch.cat(chunks, dim=0) for a, chunks in all_nulled_scores.items()}
    return baseline_out, nulled_out
