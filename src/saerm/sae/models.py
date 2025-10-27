from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


__all__ = ["SparseAutoencoder", "BatchTopKSAE"]


def _build_activation(activation: str | Callable[[torch.Tensor], torch.Tensor]) -> Callable[[torch.Tensor], torch.Tensor]:
    if callable(activation):
        raise ValueError("Custom activation functions are not supported; use 'topk'.")
    name = str(activation or "topk").lower()
    if name in {"topk", "relu"}:
        return F.relu
    raise ValueError(f"SparseAutoencoder only supports 'topk' activation (got {activation!r})")


class SparseAutoencoder(nn.Module):
    """Top-K sparse autoencoder with Matryoshka, aux-K, and batch Top-K sparsity."""

    def __init__(
        self,
        input_dim: int,
        m_total_neurons: int,
        k_active_neurons: int,
        *,
        aux_k: Optional[int] = None,
        dead_neuron_threshold_steps: int = 256,
        prefix_lengths: Optional[List[int]] = None,
        batch_topk_threshold_lr: float = 1e-2,
        activation: str = "topk",
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        if k_active_neurons <= 0:
            raise ValueError("k_active_neurons must be positive")
        if k_active_neurons > m_total_neurons:
            raise ValueError("k_active_neurons cannot exceed total neurons")

        self.input_dim = input_dim
        self.m_total_neurons = m_total_neurons
        self.k_active_neurons = k_active_neurons
        self.aux_k = aux_k if aux_k is None else max(0, min(aux_k, m_total_neurons))
        self.dead_neuron_threshold_steps = max(1, dead_neuron_threshold_steps)
        self.prefix_lengths = prefix_lengths
        self.batch_topk_threshold_lr = batch_topk_threshold_lr
        self.activation = _build_activation(activation)

        if self.prefix_lengths:
            if self.prefix_lengths[-1] != m_total_neurons:
                raise ValueError("Last prefix length must equal m_total_neurons")
            for earlier, later in zip(self.prefix_lengths[:-1], self.prefix_lengths[1:]):
                if later <= earlier:
                    raise ValueError("prefix_lengths must be strictly increasing")

        self.encoder = nn.Linear(input_dim, m_total_neurons, bias=False)
        self.decoder = nn.Linear(m_total_neurons, input_dim, bias=False)
        self.input_bias = nn.Parameter(torch.zeros(input_dim))
        self.neuron_bias = nn.Parameter(torch.zeros(m_total_neurons))

        self.register_buffer("threshold", torch.tensor(0.0))
        self.register_buffer("steps_since_activation", torch.zeros(m_total_neurons, dtype=torch.long))

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = torch.device(resolved_device)
        self.to(self._device)

    # ------------------------------------------------------------------
    # Forward & loss computation
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x_centered = x - self.input_bias
        pre_activations = self.encoder(x_centered) + self.neuron_bias
        activations, topk_idx, topk_vals = self._compute_sparse_codes(pre_activations, training=self.training)

        if self.training:
            self._update_dead_tracker(activations)

        reconstruction = self.decoder(activations) + self.input_bias

        aux_idx = aux_vals = None
        if self.training and self.aux_k and self.aux_k > 0:
            dead_mask = (self.steps_since_activation > self.dead_neuron_threshold_steps).float()
            if dead_mask.any():
                masked_pre = pre_activations * dead_mask
                aux_vals, aux_idx = torch.topk(masked_pre, self.aux_k, dim=-1)
                aux_vals = F.relu(aux_vals)

        info: Dict[str, torch.Tensor] = {
            "activations": activations,
            "pre_activations": pre_activations,
            "topk_indices": topk_idx,
            "topk_values": topk_vals,
            "aux_indices": aux_idx,
            "aux_values": aux_vals,
        }
        return reconstruction, info

    def compute_loss(
        self,
        target: torch.Tensor,
        reconstruction: torch.Tensor,
        info: Dict[str, torch.Tensor],
        aux_coef: float,
    ) -> torch.Tensor:
        activations = info["activations"]
        main_loss = self._matryoshka_loss(target, activations, reconstruction)

        if self.aux_k and info["aux_indices"] is not None and info["aux_values"] is not None:
            residual = target - reconstruction.detach()
            aux_act = torch.zeros_like(activations)
            aux_act.scatter_(-1, info["aux_indices"], info["aux_values"])
            aux_recon = self.decoder(aux_act)
            aux_loss = self._normalized_mse(aux_recon, residual)
            return main_loss + aux_coef * aux_loss

        return main_loss

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor, *, training: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        x_centered = x - self.input_bias
        pre_activations = self.encoder(x_centered) + self.neuron_bias
        activations, _, _ = self._compute_sparse_codes(pre_activations, training=training)
        return pre_activations, activations

    def initialize_weights_(self, data_sample: torch.Tensor) -> None:
        if data_sample.numel() == 0:
            return
        self.input_bias.data = torch.median(data_sample, dim=0).values
        nn.init.xavier_uniform_(self.decoder.weight)
        self.normalize_decoder_()
        self.encoder.weight.data = self.decoder.weight.t().clone()
        nn.init.zeros_(self.neuron_bias)
        self.threshold.zero_()
        self.steps_since_activation.zero_()

    def initialize_weights_kmeans_(
        self,
        data: torch.Tensor,
        *,
        num_iters: int = 50,
        tol: float = 1e-4,
        batch_size: Optional[int] = None,
        seed: Optional[int] = None,
        kmeans_plus_plus: bool = False,
    ) -> None:
        """Initialize decoder/encoder weights using K-Means centroids.

        Clusters are equal to the number of neurons (concepts). Centroids are
        normalized and written to ``decoder.weight`` (as columns), mirrored to
        the encoder, and biases/threshold trackers are reset. The input bias is
        set to the per-dimension median of the provided data.

        Args:
            data: Tensor of shape [num_examples, input_dim]. If the tensor has
                more than 2 dims, it will be flattened to [N, input_dim].
            num_iters: Maximum number of K-Means iterations.
            tol: Convergence tolerance on centroid shift (in L2, averaged per
                centroid).
            batch_size: Optional mini-batch size for streaming assignment. If
                None, uses full-batch updates (may be memory-heavy).
            seed: Optional random seed for initialization.
            kmeans_plus_plus: If True, use k-means++ initialization; otherwise
                random sample initialization.
        """
        if data is None or data.numel() == 0:
            return

        if data.dim() > 2:
            data = data.view(data.shape[0], -1)
        if data.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected data with input_dim={self.input_dim}, got {data.shape[1]}"
            )

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        num_points, dim = data.shape
        k = self.m_total_neurons
        if k <= 0:
            raise ValueError("m_total_neurons must be positive for k-means init")

        device = self._device
        dtype = self.decoder.weight.dtype

        # Compute and set input bias from data median, and center data on-the-fly
        with torch.no_grad():
            input_bias = torch.median(data, dim=0).values.to(dtype)
            self.input_bias.data = input_bias.to(device)

        # Helper to iterate mini-batches without loading everything on device
        def iter_batches(x: torch.Tensor, bs: Optional[int]):
            if bs is None or bs <= 0 or bs >= x.shape[0]:
                yield x
                return
            for start in range(0, x.shape[0], bs):
                end = min(start + bs, x.shape[0])
                yield x[start:end]

        # Initialize centroids
        with torch.no_grad():
            if kmeans_plus_plus and num_points > 0:
                # k-means++ seeding (batched distance computation)
                # Pick the first centroid uniformly at random
                first_idx = torch.randint(0, num_points, (1,)).item()
                centroids = [
                    (data[first_idx].to(device=device, dtype=dtype) - self.input_bias).to(device)
                ]
                # Precompute squared norms of data (on-the-fly per batch)
                for _ in range(1, k):
                    # Compute distances to nearest existing centroid
                    min_d2_list = []
                    for batch in iter_batches(data, batch_size):
                        xb = (batch.to(device=device, dtype=dtype) - self.input_bias)
                        # [B, C] distances to current centroids
                        d2 = []
                        for c in centroids:
                            # ||x - c||^2 = ||x||^2 + ||c||^2 - 2x·c
                            d2.append(((xb - c) ** 2).sum(dim=1, keepdim=True))
                        d2_mat = torch.cat(d2, dim=1)
                        min_d2 = d2_mat.min(dim=1).values
                        min_d2_list.append(min_d2.detach().cpu())
                    probs = torch.cat(min_d2_list, dim=0)
                    if probs.sum() <= 0:
                        next_idx = torch.randint(0, num_points, (1,)).item()
                    else:
                        next_idx = torch.multinomial(probs, 1).item()
                    centroids.append(
                        (data[next_idx].to(device=device, dtype=dtype) - self.input_bias)
                    )
                centroids = torch.stack(centroids, dim=0)  # [k, dim]
            else:
                # Random sample initialization (with replacement if needed)
                if num_points >= k:
                    perm = torch.randperm(num_points)[:k]
                else:
                    perm = torch.randint(0, num_points, (k,))
                centroids = (data[perm].to(device=device, dtype=dtype) - self.input_bias)

            # Main K-Means loop
            prev_centroids = centroids.clone()
            for _ in range(max(1, int(num_iters))):
                sums = torch.zeros((k, dim), device=device, dtype=dtype)
                counts = torch.zeros((k,), device=device, dtype=dtype)

                for batch in iter_batches(data, batch_size):
                    xb = (batch.to(device=device, dtype=dtype) - self.input_bias)
                    # Compute squared distances to centroids: [B, k]
                    # Using (x^2).sum + (c^2).sum - 2 x @ c^T
                    x2 = (xb * xb).sum(dim=1, keepdim=True)  # [B, 1]
                    c2 = (centroids * centroids).sum(dim=1).unsqueeze(0)  # [1, k]
                    dists = x2 + c2 - 2.0 * xb @ centroids.t()  # [B, k]
                    labels = torch.argmin(dists, dim=1)  # [B]

                    one_hot = F.one_hot(labels, num_classes=k).to(dtype)
                    # Update sums and counts via matrix ops for efficiency
                    sums += one_hot.t() @ xb  # [k, D]
                    counts += one_hot.sum(dim=0)  # [k]

                # Avoid division by zero; keep previous centroid where count==0
                nonzero = counts > 0
                new_centroids = centroids.clone()
                if nonzero.any():
                    new_centroids[nonzero] = sums[nonzero] / counts[nonzero].unsqueeze(1)

                # Convergence check
                shift = (new_centroids - prev_centroids).pow(2).sum(dim=1).sqrt().mean()
                centroids = new_centroids
                prev_centroids = centroids.clone()
                if shift.item() < tol:
                    break

            # Normalize centroids to unit norm and write to decoder as columns
            eps = 1e-8
            norms = centroids.norm(dim=1, keepdim=True).clamp_min(eps)
            centroids = centroids / norms

            self.decoder.weight.data = centroids.t().contiguous()
            # Keep decoder columns normalized
            self.normalize_decoder_()
            # Mirror to encoder
            self.encoder.weight.data = self.decoder.weight.t().clone()
            # Reset neuron bias, threshold, and dead tracker
            nn.init.zeros_(self.neuron_bias)
            self.threshold.zero_()
            self.steps_since_activation.zero_()

    def normalize_decoder_(self, eps: float = 1e-8) -> None:
        with torch.no_grad():
            norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(eps)
            self.decoder.weight.div_(norms)

    def adjust_decoder_gradient_(self) -> None:
        if self.decoder.weight.grad is None:
            return
        with torch.no_grad():
            proj = (self.decoder.weight * self.decoder.weight.grad).sum(dim=0, keepdim=True)
            self.decoder.weight.grad.sub_(proj * self.decoder.weight)

    def _compute_sparse_codes(
        self,
        pre_activations: torch.Tensor,
        *,
        training: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        acts = self.activation(pre_activations)
        batch_size = acts.shape[0]
        flat_acts = acts.view(-1)
        k_total = min(self.k_active_neurons * batch_size, flat_acts.numel())
        if k_total == flat_acts.numel():
            activations = acts
        else:
            vals_flat, idx_flat = torch.topk(flat_acts, k_total, dim=-1)
            activ_flat = torch.zeros_like(flat_acts)
            activ_flat.scatter_(0, idx_flat, vals_flat)
            activations = activ_flat.view_as(acts)
        if training:
            self._update_threshold_(activations)
        else:
            activations = torch.where(acts > self.threshold, acts, torch.zeros_like(acts))

        topk_vals, topk_idx = torch.topk(pre_activations, self.k_active_neurons, dim=-1)
        topk_vals = F.relu(topk_vals)
        return activations, topk_idx, topk_vals

    def _update_dead_tracker(self, activations: torch.Tensor) -> None:
        with torch.no_grad():
            self.steps_since_activation.add_(1)
            fired = (activations.sum(dim=0) > 0).nonzero(as_tuple=False).view(-1)
            if fired.numel() > 0:
                self.steps_since_activation.index_fill_(0, fired, 0)

    def _update_threshold_(self, activations: torch.Tensor) -> None:
        pos_mask = activations > 0
        if pos_mask.any():
            min_pos = activations[pos_mask].min()
            lr = self.batch_topk_threshold_lr
            self.threshold.mul_(1 - lr).add_(lr * min_pos)

    @staticmethod
    def _normalized_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(pred, target)
        baseline = F.mse_loss(target.mean(dim=0, keepdim=True).expand_as(target), target)
        if baseline <= 0:
            return mse
        return mse / (baseline + 1e-8)

    def _matryoshka_loss(self, target: torch.Tensor, activations: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
        if not self.prefix_lengths or len(self.prefix_lengths) == 1:
            return self._normalized_mse(reconstruction, target)

        losses = []
        decoder_weight = self.decoder.weight
        for end in self.prefix_lengths:
            prefix_act = activations[:, :end]
            prefix_weight = decoder_weight[:, :end]
            prefix_recon = prefix_act @ prefix_weight.t() + self.input_bias
            losses.append(self._normalized_mse(prefix_recon, target))
        return torch.stack(losses, dim=0).mean()


# Backwards compatibility ------------------------------------------------------
BatchTopKSAE = SparseAutoencoder
