"""Minimal Batch-Top-K Sparse Autoencoder.

Implements a simple SAE that, during training, selects the top-(K * B)
activations across the entire batch (BatchTopK). At evaluation time,
it falls back to per-sample Top-K for deterministic inference.

The training objective is provided via `criterion` below.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


def criterion(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    pre_codes: torch.Tensor,
    codes: torch.Tensor,
    dictionary: torch.Tensor,
) -> torch.Tensor:
    """Loss with a small revival term for dead neurons.

    Args:
        x: Input tensor (batch, input_dim)
        x_hat: Reconstruction (batch, input_dim)
        pre_codes: Pre-activation codes before sparsification (batch, m)
        codes: Sparse activations after sparsification/ReLU (batch, m)
        dictionary: Decoder weight (input_dim, m) — unused but kept for API clarity
    """
    loss = (x - x_hat).square().mean()
    is_dead = ((codes > 0).sum(dim=0) == 0).float().detach()
    reanim_loss = (pre_codes * is_dead[None, :]).mean()
    loss = loss - reanim_loss * 1e-3
    return loss


class BatchTopKSAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_neurons: int,
        k_active: int,
        *,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_neurons = num_neurons
        self.k_active = k_active
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Linear layers without bias; we keep explicit bias params
        self.encoder = nn.Linear(input_dim, num_neurons, bias=False)
        self.decoder = nn.Linear(num_neurons, input_dim, bias=False)

        self.input_bias = nn.Parameter(torch.zeros(input_dim))
        self.neuron_bias = nn.Parameter(torch.zeros(num_neurons))

        # Track dead neurons via non-activation steps counter
        self.register_buffer(
            "steps_since_activation",
            torch.zeros(num_neurons, dtype=torch.long),
        )

        # Optional threshold buffer (not used for inference here, but may be useful)
        self.register_buffer("threshold", torch.tensor(0.0))

        self.to(self.device)

    # --------------------------- initialization utils ---------------------------
    @torch.no_grad()
    def initialize_weights_(self, data_sample: torch.Tensor) -> None:
        self.input_bias.copy_(torch.median(data_sample, dim=0).values)
        nn.init.xavier_uniform_(self.decoder.weight)
        self.normalize_decoder_()
        self.encoder.weight.copy_(self.decoder.weight.t())
        nn.init.zeros_(self.neuron_bias)

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        self.decoder.weight.div_(
            self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-12)
        )

    @torch.no_grad()
    def _update_threshold_(self, activ: torch.Tensor, lr: float = 1e-2) -> None:
        # EMA towards the smallest positive activation in the batch
        pos = activ > 0
        if pos.any():
            min_pos = activ[pos].min()
            self.threshold.mul_(1 - lr).add_(lr * min_pos)

    def adjust_decoder_gradient_(self) -> None:
        if self.decoder.weight.grad is None:
            return
        with torch.no_grad():
            proj = (self.decoder.weight * self.decoder.weight.grad).sum(
                dim=0, keepdim=True
            )
            self.decoder.weight.grad.sub_(proj * self.decoder.weight)

    # --------------------------------- forward ---------------------------------
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x_centered = x - self.input_bias
        pre_codes = self.encoder(x_centered) + self.neuron_bias  # (B, M)

        if self.training:  # BatchTopK over the entire batch
            acts = F.relu(pre_codes)
            batch_size, m = acts.shape
            k_total = min(self.k_active * batch_size, batch_size * m)
            flat = acts.reshape(-1)
            vals, idx = torch.topk(flat, k_total, dim=0)
            flat_sparse = torch.zeros_like(flat)
            flat_sparse.scatter_(0, idx, vals)
            codes = flat_sparse.view_as(acts)
            self._update_threshold_(codes)
        else:  # Per-sample TopK at inference for deterministic K-sparsity
            vals, idx = torch.topk(pre_codes, k=self.k_active, dim=-1)
            vals = F.relu(vals)
            codes = torch.zeros_like(pre_codes)
            codes.scatter_(-1, idx, vals)

        # Update dead-neuron tracker
        with torch.no_grad():
            self.steps_since_activation.add_(1)
            fired = (codes.sum(dim=0) > 0).nonzero(as_tuple=False).squeeze(-1)
            if fired.numel() > 0:
                self.steps_since_activation.index_fill_(0, fired, 0)

        x_hat = self.decoder(codes) + self.input_bias
        info = {
            "pre_codes": pre_codes,
            "codes": codes,
            "dictionary": self.decoder.weight,
        }
        return x_hat, info

    # ---------------------------------- train ----------------------------------
    def fit(
        self,
        X_train: torch.Tensor,
        X_val: Optional[torch.Tensor] = None,
        *,
        batch_size: int = 512,
        learning_rate: float = 5e-4,
        n_epochs: int = 100,
        patience: int = 5,
        clip_grad: Optional[float] = 1.0,
        show_progress: bool = True,
    ) -> Dict[str, list]:
        train_loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=True)
        val_loader = (
            DataLoader(TensorDataset(X_val), batch_size=batch_size) if X_val is not None else None
        )

        # Initialize using the full training tensor on the target device
        self.initialize_weights_(X_train.to(self.device))

        opt = torch.optim.Adam(self.parameters(), lr=learning_rate)
        best_val = float("inf")
        patience_ctr = 0
        history = {"train_loss": [], "val_loss": []}

        epoch_iter = tqdm(range(n_epochs), desc="Epochs") if show_progress else range(n_epochs)
        for _ in epoch_iter:
            self.train()
            train_losses = []
            train_batch_iter = tqdm(train_loader, desc="Train", leave=False) if show_progress else train_loader
            for (batch_x,) in train_batch_iter:
                batch_x = batch_x.to(self.device)
                x_hat, info = self(batch_x)
                loss = criterion(batch_x, x_hat, info["pre_codes"], info["codes"], info["dictionary"])

                opt.zero_grad()
                loss.backward()
                self.adjust_decoder_gradient_()
                if clip_grad is not None:
                    nn.utils.clip_grad_norm_(self.parameters(), clip_grad)
                opt.step()
                self.normalize_decoder_()
                train_losses.append(loss.item())

            avg_train = float(np.mean(train_losses)) if train_losses else 0.0
            history["train_loss"].append(avg_train)

            if val_loader is not None:
                self.eval()
                val_losses = []
                with torch.no_grad():
                    val_batch_iter = tqdm(val_loader, desc="Val", leave=False) if show_progress else val_loader
                    for (batch_x,) in val_batch_iter:
                        batch_x = batch_x.to(self.device)
                        x_hat, info = self(batch_x)
                        vloss = criterion(batch_x, x_hat, info["pre_codes"], info["codes"], info["dictionary"])
                        val_losses.append(vloss.item())
                avg_val = float(np.mean(val_losses)) if val_losses else 0.0
                history["val_loss"].append(avg_val)

                improved = avg_val < best_val - 1e-8
                best_val = min(best_val, avg_val)
                if not improved:
                    patience_ctr += 1
                    if patience_ctr >= patience:
                        break
                else:
                    patience_ctr = 0

            if show_progress:
                # Update progress info succinctly
                postfix = {"train": f"{avg_train:.4f}"}
                if val_loader is not None:
                    postfix["val"] = f"{history['val_loss'][-1]:.4f}"
                epoch_iter.set_postfix(postfix)

        return history

    # ----------------------------- batched inference ----------------------------
    @torch.no_grad()
    def get_activations(self, inputs, batch_size: int = 8192, show_progress: bool = True) -> np.ndarray:
        self.eval()

        if isinstance(inputs, list):
            inputs = torch.tensor(inputs, dtype=torch.float)
        elif isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).float()
        elif not isinstance(inputs, torch.Tensor):
            raise TypeError("inputs must be a list, numpy array, or torch tensor")
        if inputs.dtype != torch.float:
            inputs = inputs.float()

        total = inputs.shape[0]
        all_codes = []
        rng = tqdm(range(0, total, batch_size), desc=f"Activations (batch={batch_size})") if show_progress else range(0, total, batch_size)
        for i in rng:
            batch = inputs[i : i + batch_size].to(self.device)
            x_hat, info = self(batch)
            all_codes.append(info["codes"].cpu())
        return torch.cat(all_codes, dim=0).numpy()


