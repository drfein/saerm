from __future__ import annotations

from typing import Tuple

import torch


@torch.no_grad()
def compute_sae_metrics(mounted_sae, model_inputs, *, sample_cap: int | None = None) -> Tuple[float, float]:
    """Compute sparsity and R^2 reconstruction quality for a MountedSAE.

    Args:
        mounted_sae: An instance of saerm.sae.mounted.MountedSAE (with hook registered).
        model_inputs: Inputs expected by the mounted_sae.base_model (tensor, dict of tensors, or list/tuple).
        sample_cap: Optional limit on the number of examples to use.

    Returns:
        (sparsity, r2): tuple of floats
    """
    # Optionally subsample
    if sample_cap is not None:
        if isinstance(model_inputs, torch.Tensor):
            model_inputs = model_inputs[:sample_cap]
        elif isinstance(model_inputs, dict):
            model_inputs = {k: (v[:sample_cap] if isinstance(v, torch.Tensor) else v) for k, v in model_inputs.items()}
        elif isinstance(model_inputs, (list, tuple)):
            model_inputs = model_inputs[:sample_cap]

    # 1) Run base model once to capture activations via the hook
    if isinstance(model_inputs, dict):
        _ = mounted_sae.base_model(**model_inputs)
    else:
        _ = mounted_sae.base_model(model_inputs)
    feats = mounted_sae._pop_cached_activation()

    # 2) Forward through SAE to get reconstruction and codes
    x_hat, info = mounted_sae.sae(feats.to(mounted_sae.sae.device))  # type: ignore[union-attr]

    # 3) Sparsity
    codes = info["codes"].detach().cpu()
    density = (codes > 0).float().mean().item()
    sparsity = 1.0 - density

    # 4) R^2
    x = feats.detach().cpu()
    xh = x_hat.detach().cpu()
    sse = ((x - xh) ** 2).sum().item()
    mu = x.mean(dim=0, keepdim=True)
    sst = ((x - mu) ** 2).sum().item() + 1e-12
    r2 = 1.0 - (sse / sst)

    return sparsity, r2


__all__ = ["compute_sae_metrics"]


