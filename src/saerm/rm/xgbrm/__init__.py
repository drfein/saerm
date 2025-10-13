from __future__ import annotations

from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple
import os

import torch
import torch.nn as nn

try:
    import xgboost as xgb  # type: ignore
except Exception as _xgb_err:  # pragma: no cover
    xgb = None  # type: ignore

from ..base import BaseRewardModel, register_reward_model
from ...process_data.base import BasePreferenceDataset


@register_reward_model("xgbrm")
class XGBSAERM(BaseRewardModel):
    """XGBoost-based reward model on top of SAE codes.

    Trains an XGBoost ranker (pairwise) on standardized SAE codes.
    The model outputs a scalar score s(z) for a single text's SAE code z.
    """

    def __init__(
        self,
        *,
        sae: Optional[nn.Module] = None,
        encode_fn: Optional[Callable[[Sequence[str]], object]] = None,
        device: Optional[str] = None,
        encode_batch_size: int = 32,
        xgb_params: Optional[dict] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sae = sae
        self.encode_fn = encode_fn
        self.encode_batch_size = int(encode_batch_size)

        self._mu: Optional[torch.Tensor] = None
        self._sig: Optional[torch.Tensor] = None

        self._xgb_model: Optional[Any] = None  # XGBRanker or Booster-like
        # Default XGB params; can be overridden at training time
        self._xgb_params = xgb_params or {
            "objective": "rank:pairwise",
            "learning_rate": 0.05,
            "max_depth": 8,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_estimators": 500,
            "tree_method": "hist",
        }

    # ------------------------------- helpers --------------------------------
    def _require_components(self) -> None:
        if self.sae is None:
            raise RuntimeError("XGBSAERM requires an SAE instance (sae=...) to be set.")
        if self.encode_fn is None:
            raise RuntimeError(
                "XGBSAERM requires an encode_fn(list[str])->torch.Tensor to map texts to SAE inputs."
            )
        if xgb is None:  # pragma: no cover
            raise RuntimeError("xgboost is not installed. Please `pip install xgboost`." )

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
            _xh, info = self.sae(batch_inputs)  # type: ignore[misc]
            codes = info.get("codes") if isinstance(info, dict) else None
            if codes is None:
                raise RuntimeError("SAE forward must return dict with 'codes' tensor")
            codes_list.append(codes.detach().to(self.device))
        if not codes_list:
            return torch.zeros(0, 0, device=self.device)
        return torch.cat(codes_list, dim=0)

    def _standardize(self, Z: torch.Tensor) -> torch.Tensor:
        if self._mu is None or self._sig is None:
            return Z
        return (Z - self._mu) / self._sig

    # --------------------------------- API -----------------------------------
    def reward(self, body: str, prompt: Optional[str] = None) -> float:
        if self._xgb_model is None:
            raise RuntimeError("XGBSAERM not trained: XGBoost model is missing.")
        self._require_components()
        Z = self._codes_from_texts([body])  # (1, M)
        Z = self._standardize(Z)
        X = Z.detach().cpu().numpy()
        # XGBRanker/XGBClassifier both expose predict(X)
        pred = self._xgb_model.predict(X)
        # Ensure scalar float
        val = float(pred.reshape(-1)[0])
        return val

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

        # Standardize using chosen stats
        self._mu = Z_ch.mean(dim=0)
        self._sig = Z_ch.std(dim=0).clamp_min(1e-8)
        Z_ch = self._standardize(Z_ch)
        Z_rj = self._standardize(Z_rj)

        # Build interleaved items per pair: [ch0, rj0, ch1, rj1, ...]
        n_pairs = Z_ch.size(0)
        interleaved = torch.stack((Z_ch, Z_rj), dim=1).reshape(n_pairs * 2, -1)
        X = interleaved.detach().cpu().numpy()
        y = torch.tensor([1, 0], device=self.device).repeat(n_pairs).detach().cpu().numpy()
        group = [2] * n_pairs

        # Small validation slice from the tail (by whole pairs)
        n_val_pairs = max(1000, int(0.05 * n_pairs)) if n_pairs >= 2000 else max(50, int(0.02 * n_pairs))
        n_val_pairs = min(n_val_pairs, n_pairs // 10) if n_pairs >= 20 else 0
        if n_val_pairs > 0:
            n_train_pairs = n_pairs - n_val_pairs
            X_tr = X[: 2 * n_train_pairs]
            y_tr = y[: 2 * n_train_pairs]
            grp_tr = [2] * n_train_pairs
            X_va = X[2 * n_train_pairs :]
            y_va = y[2 * n_train_pairs :]
            grp_va = [2] * n_val_pairs
        else:
            X_tr, y_tr, grp_tr = X, y, group
            X_va, y_va, grp_va = None, None, None

        model = xgb.XGBRanker(**self._xgb_params)
        if X_va is not None:
            model.fit(
                X_tr,
                y_tr,
                group=grp_tr,
                eval_set=[(X_va, y_va)],
                eval_group=[grp_va],
                verbose=False,
            )
        else:
            model.fit(X_tr, y_tr, group=grp_tr, verbose=False)

        self._xgb_model = model

    def train_from_codes(self, Z_ch: torch.Tensor, Z_rj: torch.Tensor) -> None:
        """Train directly from precomputed SAE codes for chosen/rejected (tensors)."""
        self._require_components()
        # Standardize using chosen stats
        self._mu = Z_ch.mean(dim=0)
        self._sig = Z_ch.std(dim=0).clamp_min(1e-8)
        Z_ch = self._standardize(Z_ch)
        Z_rj = self._standardize(Z_rj)
        n_pairs = Z_ch.size(0)
        interleaved = torch.stack((Z_ch, Z_rj), dim=1).reshape(n_pairs * 2, -1)
        X = interleaved.detach().cpu().numpy()
        y = torch.tensor([1, 0], device=self.device).repeat(n_pairs).detach().cpu().numpy()
        group = [2] * n_pairs
        model = xgb.XGBRanker(**self._xgb_params)
        model.fit(X, y, group=group, verbose=False)
        self._xgb_model = model

    @classmethod
    def load(cls, path: str, config: Mapping[str, Any]) -> "XGBSAERM":
        device = config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        sae = config.get("sae")
        encode_fn = config.get("encode_fn")
        encode_batch_size = int(config.get("encode_batch_size", 32))
        xgb_params = config.get("xgb_params")

        inst = cls(sae=sae, encode_fn=encode_fn, device=device, encode_batch_size=encode_batch_size, xgb_params=xgb_params)

        # Load xgb model if present
        xgb_path = os.path.join(path, "xgb.json")
        if os.path.exists(xgb_path):
            if xgb is None:  # pragma: no cover
                raise RuntimeError("xgboost is not installed. Please `pip install xgboost`.")
            model = xgb.XGBRanker()
            model.load_model(xgb_path)
            inst._xgb_model = model
        # Load stats (mu/sig)
        stats_path = os.path.join(path, "stats.pt")
        if os.path.exists(stats_path):
            stats = torch.load(stats_path, map_location=device)
            inst._mu = stats.get("mu")
            inst._sig = stats.get("sig")

        return inst

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        if self._xgb_model is not None:
            # Save model in JSON format
            self._xgb_model.save_model(os.path.join(path, "xgb.json"))
        if self._mu is not None and self._sig is not None:
            torch.save({"mu": self._mu, "sig": self._sig}, os.path.join(path, "stats.pt"))


__all__ = ["XGBSAERM"]



