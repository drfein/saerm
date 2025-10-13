from ..base import BaseRewardModel, register_reward_model
from typing import Any, Mapping, Optional
import os


@register_reward_model("btrm")
class BTRM(BaseRewardModel):
    """BatchTopK/LLM-based Reward Model loaded from Hugging Face or local path.

    This implementation mirrors the single-text scoring used in the notebook
    (e.g., using a sequence classification head). For `reward`, if no prompt is
    provided, the score is computed on `body` alone; if a prompt is given, a
    simple joined template is used.
    """

    def __init__(self, model: Any, tokenizer: Any, max_length: int = 512, device: str = "cpu") -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._max_length = int(max_length)
        self._device = device

    @staticmethod
    def _compose_input(body: str, prompt: Optional[str]) -> str:
        if prompt is None or str(prompt).strip() == "":
            return body
        # Minimal prompt-response join; may be improved later if needed
        return f"Prompt:\n{prompt}\n\nResponse:\n{body}"

    @staticmethod
    def _scalar_score(logits: Any) -> Any:
        """Convert model logits to a scalar score consistent with notebook usage.

        - If shape is [*, 1], squeeze the last dim
        - If shape is [*, 2], use logit[0] - logit[1]
        - Else, mean across last dim
        """
        import torch

        if not hasattr(logits, "ndim"):
            return logits
        if logits.ndim == 0:
            return logits
        if logits.size(-1) == 1:
            return logits.squeeze(-1)
        if logits.size(-1) == 2:
            return logits[..., 0] - logits[..., 1]
        return logits.mean(dim=-1)

    def reward(self, body: str, prompt: Optional[str] = None) -> float:
        import torch

        text = self._compose_input(body=body, prompt=prompt)
        enc = self._tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self._device) for k, v in enc.items()}
        self._model.eval()
        with torch.no_grad():
            out = self._model(**enc)
            score = self._scalar_score(out.logits)
        # Ensure Python float
        return float(score.reshape(-1)[0].item())

    def train(self, dataset: Any) -> None:
        raise NotImplementedError("BTRM training is not implemented in this package module.")

    @classmethod
    def load(cls, path: str, config: Mapping[str, Any]) -> "BTRM":
        """Load from a local directory or Hugging Face repo id.

        Args:
            path: Filesystem path or Hugging Face repo id (e.g., "user/repo").
            config: Optional fields:
                - hf_token: str, Hugging Face access token for gated models
                - max_length: int, tokenizer max_length (default 512)
                - device: "cuda" | "cpu" | device string; auto-detected if omitted
                - trust_remote_code: bool, passed to transformers loaders
        """
        # Lazy imports to avoid heavy deps at module import time
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        hf_token = config.get("hf_token") or os.environ.get("HF_TOKEN") or None
        max_length = int(config.get("max_length", 512) or 512)
        trust_remote_code = bool(config.get("trust_remote_code", False))

        # Auto-select device
        if isinstance(config.get("device"), str):
            device = config.get("device")  # type: ignore[assignment]
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load tokenizer
        tok_kwargs = {"use_fast": True, "trust_remote_code": trust_remote_code}
        if hf_token:
            tok_kwargs["token"] = hf_token
        tokenizer = AutoTokenizer.from_pretrained(path, **tok_kwargs)
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Load sequence classification model; do not override num_labels if saved
        mdl_kwargs = {"trust_remote_code": trust_remote_code}
        if hf_token:
            mdl_kwargs["token"] = hf_token
        # Use bf16 if available (A100+), else default
        try:
            bf16_ok = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
            if bf16_ok:
                mdl_kwargs["dtype"] = torch.bfloat16
        except Exception:
            pass

        model = AutoModelForSequenceClassification.from_pretrained(path, **mdl_kwargs)
        model.config.pad_token_id = tokenizer.pad_token_id
        model.to(device)

        return cls(model=model, tokenizer=tokenizer, max_length=max_length, device=device)

__all__ = ["BTRM"]


