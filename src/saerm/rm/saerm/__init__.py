from __future__ import annotations

import os
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseRewardModel, register_reward_model
from ...process_data.base import BasePreferenceDataset


def _bt_loss(s_ch: torch.Tensor, s_rj: torch.Tensor) -> torch.Tensor:
    return F.softplus(-(s_ch - s_rj)).mean()


@register_reward_model("saerm")
class SAERM(BaseRewardModel):
    """SAE-based reward model with a linear BT head over SAE codes.

    Usage:
        - Provide an SAE instance (e.g., BatchTopKSAE or MountedSAE) and an
          `encode_fn` that maps a list of texts to model inputs expected by the
          SAE/base model. This may be a tensor, numpy array, or a dict of tensors
          (e.g., tokenizer outputs for a transformer).
        - Call `train(dataset)` to fit the linear head weights using BT loss.
        - Call `reward(text)` to obtain a scalar score.
    """

    def __init__(
        self,
        *,
        sae: Optional[nn.Module] = None,
        encode_fn: Optional[Callable[[Sequence[str]], object]] = None,
        device: Optional[str] = None,
        encode_batch_size: int = 32,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sae = sae
        self.encode_fn = encode_fn
        self.encode_batch_size = int(encode_batch_size)

        self._w: Optional[torch.Tensor] = None  # linear head weights (M,)
        self._mu: Optional[torch.Tensor] = None  # codes mean (M,)
        self._sig: Optional[torch.Tensor] = None  # codes std (M,)

    # ------------------------------- helpers --------------------------------
    def _require_components(self) -> None:
        if self.sae is None:
            raise RuntimeError("SAERM requires an SAE instance (sae=...) to be set.")
        if self.encode_fn is None:
            raise RuntimeError(
                "SAERM requires an encode_fn(list[str])->torch.Tensor to map texts to SAE inputs."
            )

    def _move_to_device(self, obj: object) -> object:
        if isinstance(obj, torch.Tensor):
            return obj.to(self.device)
        if isinstance(obj, dict):
            return {k: self._move_to_device(v) for k, v in obj.items()}  # type: ignore[dict-item]
        if isinstance(obj, (list, tuple)):
            moved = [self._move_to_device(v) for v in obj]
            return type(obj)(moved)  # type: ignore[call-arg]
        return obj

    @torch.no_grad()
    def _encode_texts(self, texts: Sequence[str]):
        data = self.encode_fn(texts)  # type: ignore[misc]
        return self._move_to_device(data)

    @torch.no_grad()
    def _codes_from_texts(self, texts: Sequence[str]) -> torch.Tensor:
        assert self.sae is not None
        self.sae.eval()
        codes_list: List[torch.Tensor] = []
        for i in range(0, len(texts), self.encode_batch_size):
            batch_inputs = self._encode_texts(texts[i : i + self.encode_batch_size])
            x_hat, info = self.sae(batch_inputs)  # type: ignore[misc]
            codes = info.get("codes") if isinstance(info, dict) else None
            if codes is None:
                raise RuntimeError("SAE forward must return dict with 'codes' tensor")
            codes_list.append(codes.detach().to(self.device))
        if not codes_list:
            return torch.zeros(0, getattr(self._w, "numel", lambda: 0)(), device=self.device)
        return torch.cat(codes_list, dim=0)

    def _standardize(self, Z: torch.Tensor) -> torch.Tensor:
        assert self._mu is not None and self._sig is not None
        return (Z - self._mu) / self._sig

    # --------------------------------- API -----------------------------------
    def reward(self, body: str, prompt: Optional[str] = None) -> float:
        if self._w is None:
            raise RuntimeError("SAERM not trained: linear head weights are missing.")
        self._require_components()
        Z = self._codes_from_texts([body])  # (1, M)
        if self._mu is not None and self._sig is not None:
            Z = self._standardize(Z)
        score = float((Z @ self._w).squeeze().item())
        return score

    def train(self, dataset: Any) -> None:
        # Expect a BasePreferenceDataset or similar exposing text pairs
        if isinstance(dataset, BasePreferenceDataset):
            pairs = dataset.as_text_tuples(split="train")
        elif hasattr(dataset, "as_text_tuples"):
            pairs = dataset.as_text_tuples(split="train")  # type: ignore[assignment]
        else:
            raise TypeError("Expected a BasePreferenceDataset or object with as_text_tuples(split=...) method")

        self._require_components()
        chosen = [c for c, _ in pairs]
        rejected = [r for _, r in pairs]

        # Codes via SAE
        Z_ch = self._codes_from_texts(chosen)  # (N, M)
        Z_rj = self._codes_from_texts(rejected)  # (N, M)

        # Standardize using chosen stats (simple and effective)
        self._mu = Z_ch.mean(dim=0)
        self._sig = Z_ch.std(dim=0).clamp_min(1e-8)
        Z_ch = self._standardize(Z_ch)
        Z_rj = self._standardize(Z_rj)

        # Linear BT head
        m = Z_ch.shape[1]
        w = torch.zeros(m, device=self.device, requires_grad=True)

        lr = 1e-2
        wd = 1e-4
        epochs = 10
        batch_size = 4096
        opt = torch.optim.AdamW([w], lr=lr, weight_decay=wd)

        n = Z_ch.size(0)
        for _ in range(epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                zc = Z_ch[idx]
                zr = Z_rj[idx]
                s_ch = zc @ w
                s_rj = zr @ w
                loss = _bt_loss(s_ch, s_rj)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        self._w = w.detach()

    @classmethod
    def load(cls, path: str, config: Mapping[str, Any]) -> "SAERM":
        # Best-effort loader: optional SAE and head weights
        device = config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        sae = config.get("sae")  # allow passing a prebuilt SAE instance
        encode_fn = config.get("encode_fn")

        inst = cls(sae=sae, encode_fn=encode_fn, device=device)

        head_path = os.path.join(path, "head.pt")
        stats_path = os.path.join(path, "stats.pt")
        if os.path.exists(head_path):
            head = torch.load(head_path, map_location=device)
            inst._w = head.get("w") if isinstance(head, dict) else head
        if os.path.exists(stats_path):
            stats = torch.load(stats_path, map_location=device)
            inst._mu = stats.get("mu")
            inst._sig = stats.get("sig")
        return inst

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        if self._w is not None:
            torch.save({"w": self._w}, os.path.join(path, "head.pt"))
        if self._mu is not None and self._sig is not None:
            torch.save({"mu": self._mu, "sig": self._sig}, os.path.join(path, "stats.pt"))


__all__ = ["SAERM"]

