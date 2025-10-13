from typing import Callable, Dict, Optional, Tuple, Union, Sequence, Any
import contextlib
from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from .batchtopk_sae import BatchTopKSAE, criterion
import os
import json


def _resolve_module(root: nn.Module, path: str) -> nn.Module:
    cur: nn.Module = root
    for name in path.split("."):
        if not hasattr(cur, name):
            raise AttributeError(f"Module '{type(cur).__name__}' has no submodule/attr '{name}' in path '{path}'")
        cur = getattr(cur, name)
        if not isinstance(cur, nn.Module):
            raise TypeError(f"Attribute '{name}' in path '{path}' is not a torch.nn.Module")
    return cur


class MountedSAE(nn.Module):
    """Mount a BatchTopKSAE on top of a given layer in a base model.

    The forward pass first runs the base model to produce layer activations, then
    feeds those activations through the SAE. Training the SAE can be done by passing
    in raw inputs for the base model; gradients do not flow into the base model.

    Args:
        base_model: The underlying model whose layer activations are encoded.
        layer: Layer spec on which to mount. Either a dotted path from base_model, or the module itself.
        sae: Optionally provide a prebuilt BatchTopKSAE. If None, it is lazily
             created on first forward using the observed activation dimensionality.
        num_neurons: Required if sae is None. Number of SAE neurons.
        k_active: Required if sae is None. Number of active units per sample (eval) / per-batch TopK (train).
        device: Device for the SAE. Defaults to the base model's device.
        flatten: If True, flattens the layer output to shape (B, -1) before SAE.
        activation_transform: Optional callable to transform the raw layer output
            into a tensor suitable for the SAE (applied before flattening).
    """

    def __init__(
        self,
        base_model: nn.Module,
        layer: Union[str, nn.Module],
        *,
        sae: Optional[BatchTopKSAE] = None,
        num_neurons: Optional[int] = None,
        k_active: Optional[int] = None,
        device: Optional[str] = None,
        flatten: bool = True,
        activation_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.flatten = flatten
        self.activation_transform = activation_transform
        self._cached_activation: Optional[torch.Tensor] = None

        # Resolve target layer
        if isinstance(layer, str):
            self.target_layer = _resolve_module(base_model, layer)
        elif isinstance(layer, nn.Module):
            self.target_layer = layer
        else:
            raise TypeError("layer must be a dotted path or nn.Module")

        # Detect base model device
        try:
            self.base_device = next(self.base_model.parameters()).device
        except StopIteration:
            self.base_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.sae: Optional[BatchTopKSAE] = sae
        self._num_neurons = num_neurons
        self._k_active = k_active
        self._sae_device = device  # resolved lazily if None

        # Make sure base model is frozen (no gradients) for SAE training
        self.freeze_base_model_()

        # Register forward hook to capture layer activations
        self._hook_handle = self.target_layer.register_forward_hook(self._forward_hook)

    # ---------------------------- lifecycle utilities ----------------------------
    def freeze_base_model_(self) -> None:
        for p in self.base_model.parameters():
            p.requires_grad = False
        self.base_model.eval()

    def unfreeze_base_model_(self) -> None:
        for p in self.base_model.parameters():
            p.requires_grad = True

    def remove_hook_(self) -> None:
        if hasattr(self, "_hook_handle") and self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    # ------------------------------ hook and helpers -----------------------------
    def _extract_tensor_from_output(self, output: Any) -> torch.Tensor:
        """Best-effort extraction of a tensor from an arbitrary model output.

        Preference order:
        1) hidden_states[-1]
        2) last_hidden_state
        3) logits
        4) first tensor found in a (nested) tuple/list/dict
        """
        # Direct tensor
        if isinstance(output, torch.Tensor):
            return output

        # Objects with attrs (e.g., HF ModelOutput)
        # hidden_states may be a tuple[list] of tensors
        hs = getattr(output, "hidden_states", None)
        if hs is not None:
            if isinstance(hs, (list, tuple)) and len(hs) > 0:
                last = hs[-1]
                if isinstance(last, torch.Tensor):
                    return last
            if isinstance(hs, torch.Tensor):
                return hs

        lhm = getattr(output, "last_hidden_state", None)
        if isinstance(lhm, torch.Tensor):
            return lhm

        logits = getattr(output, "logits", None)
        if isinstance(logits, torch.Tensor):
            return logits

        # Mapping-like output
        if isinstance(output, Mapping):
            # Prefer common keys if present
            if "hidden_states" in output:
                hs_val = output["hidden_states"]
                if isinstance(hs_val, (list, tuple)) and len(hs_val) > 0 and isinstance(hs_val[-1], torch.Tensor):
                    return hs_val[-1]
                if isinstance(hs_val, torch.Tensor):
                    return hs_val
            if "last_hidden_state" in output and isinstance(output["last_hidden_state"], torch.Tensor):
                return output["last_hidden_state"]
            if "logits" in output and isinstance(output["logits"], torch.Tensor):
                return output["logits"]
            # Fallback: first tensor value
            for v in output.values():
                if isinstance(v, torch.Tensor):
                    return v
                if isinstance(v, (list, tuple)):
                    for item in reversed(v):
                        if isinstance(item, torch.Tensor):
                            return item

        # Sequence output: choose the last tensor if present
        if isinstance(output, (list, tuple)):
            for item in reversed(output):
                if isinstance(item, torch.Tensor):
                    return item
                if isinstance(item, (list, tuple)):
                    for sub in reversed(item):
                        if isinstance(sub, torch.Tensor):
                            return sub

        raise TypeError("Unable to extract tensor from model output; provide an activation_transform to specify how to handle outputs.")

    def _forward_hook(self, _module: nn.Module, _inputs: Tuple[torch.Tensor, ...], output: Any) -> None:
        # Support transforms that accept raw model outputs or already-extracted tensors
        activ: torch.Tensor
        if self.activation_transform is not None:
            used = False
            try:
                maybe = self.activation_transform(output)
                if isinstance(maybe, torch.Tensor):
                    activ = maybe
                    used = True
            except Exception:
                # Ignore and try tensor path
                used = False
            if not used:
                activ = self._extract_tensor_from_output(output)
                try:
                    maybe2 = self.activation_transform(activ)
                    if isinstance(maybe2, torch.Tensor):
                        activ = maybe2
                except Exception:
                    # Keep activ as extracted tensor
                    pass
        else:
            activ = self._extract_tensor_from_output(output)
        if self.flatten:
            if activ.dim() == 1:
                activ = activ.unsqueeze(0)
            else:
                activ = activ.view(activ.shape[0], -1)
        self._cached_activation = activ.detach()

    def _pop_cached_activation(self) -> torch.Tensor:
        if self._cached_activation is None:
            raise RuntimeError("No activation captured. Ensure the hook is registered and the base model forward was executed.")
        activ = self._cached_activation
        self._cached_activation = None
        return activ

    def _ensure_sae_initialized(self, feature_batch: torch.Tensor) -> None:
        if self.sae is not None:
            return
        if self._num_neurons is None or self._k_active is None:
            raise ValueError("num_neurons and k_active are required when sae is not provided")
        input_dim = feature_batch.shape[-1]
        sae_device = self._sae_device or ("cuda" if torch.cuda.is_available() else str(self.base_device))
        self.sae = BatchTopKSAE(
            input_dim=input_dim,
            num_neurons=self._num_neurons,
            k_active=self._k_active,
            device=sae_device,
        )
        # Initialize SAE with a sample
        with torch.no_grad():
            self.sae.initialize_weights_(feature_batch.to(self.sae.device))

    # ---------------------------------- forward ---------------------------------
    def _move_to_device(self, obj: object, device: torch.device) -> object:
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        if isinstance(obj, Mapping):
            return {k: self._move_to_device(v, device) for k, v in obj.items()}  # type: ignore[dict-item]
        if isinstance(obj, (list, tuple)):
            moved = [self._move_to_device(v, device) for v in obj]
            return type(obj)(moved)  # type: ignore[call-arg]
        return obj

    def forward(self, x, *, return_base_output: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        with torch.no_grad():
            base_inputs = self._move_to_device(x, self.base_device)
            if isinstance(base_inputs, Mapping):
                base_out = self.base_model(**base_inputs)
            else:
                base_out = self.base_model(base_inputs)
        feat = self._pop_cached_activation()
        self._ensure_sae_initialized(feat)
        x_hat, info = self.sae(feat.to(self.sae.device))  # type: ignore[arg-type]
        if return_base_output:
            info = {**info, "base_output": base_out}
        return x_hat, info

    # -------------------------- feature extraction utils --------------------------
    @torch.no_grad()
    def extract_features_to_dir(
        self,
        X,
        out_dir: str,
        *,
        batch_size: int = 256,
        shard_size: int = 20000,
        dtype: str = "float16",
        show_progress: bool = True,
        batch_collate: Optional[Callable[[Sequence[object]], object]] = None,
        amp_dtype: Optional[str] = None,
    ) -> dict:
        os.makedirs(out_dir, exist_ok=True)
        from torch.utils.data import DataLoader, TensorDataset
        from collections.abc import Mapping

        # ------------------------- resume: read manifest -------------------------
        resume_manifest = None
        resume_rows = 0
        next_shard_idx = 0
        feat_dim_resume: Optional[int] = None
        manifest_path = os.path.join(out_dir, "manifest.json")
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r") as f:
                    resume_manifest = json.load(f)
                shards_meta_resume = resume_manifest.get("shards", [])
                resume_rows = int(sum(int(s.get("count", 0)) for s in shards_meta_resume))
                next_shard_idx = int(len(shards_meta_resume))
                feat_dim_resume = int(resume_manifest.get("dim", 0)) or None
            except Exception:
                resume_manifest = None
                resume_rows = 0
                next_shard_idx = 0

        if isinstance(X, torch.Tensor):
            loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=False)
        else:
            from torch.utils.data import Dataset

            class ItemsDataset(Dataset):  # type: ignore[type-arg]
                def __init__(self, data):
                    self.data = data

                def __len__(self):
                    d = self.data
                    if isinstance(d, torch.Tensor):
                        return d.shape[0]
                    if isinstance(d, Mapping):
                        for v in d.values():
                            if isinstance(v, torch.Tensor):
                                return v.shape[0]
                        raise TypeError("Dict values must include at least one tensor to infer length")
                    if isinstance(d, (list, tuple)):
                        return len(d)
                    raise TypeError("Unsupported dataset type")

                def __getitem__(self, idx):
                    d = self.data
                    if isinstance(d, torch.Tensor):
                        return d[idx]
                    if isinstance(d, Mapping):
                        return {k: (v[idx] if isinstance(v, torch.Tensor) else v) for k, v in d.items()}
                    if isinstance(d, (list, tuple)):
                        return d[idx]
                    raise TypeError("Unsupported dataset type")

            loader = DataLoader(ItemsDataset(X), batch_size=batch_size, shuffle=False)

        # If everything is already cached, fast-exit
        try:
            expected_total = len(loader.dataset)  # type: ignore[attr-defined]
        except Exception:
            expected_total = None
        if expected_total is not None and resume_rows >= expected_total:
            if show_progress:
                tqdm.write("Resume complete: all rows already cached; nothing to do")
            return resume_manifest or {
                "total": int(resume_rows),
                "dim": int(feat_dim_resume or 0),
                "dtype": dtype,
                "shard_size": int(shard_size),
                "shards": [] if resume_manifest is None else list(resume_manifest.get("shards", [])),
            }

        it = tqdm(loader, desc="Extract feats", leave=False) if show_progress else loader
        dtype_map = {"float16": torch.float16, "float32": torch.float32, "bf16": torch.bfloat16}
        store_dtype = dtype_map.get(dtype, torch.float16)

        total = 0
        shard_idx = next_shard_idx
        shard_rows = []
        shard_row_count = 0
        feat_dim = feat_dim_resume
        shards_meta = [] if resume_manifest is None else list(resume_manifest.get("shards", []))

        # ----------------------------- skip rows to resume -----------------------------
        rows_to_skip = resume_rows
        for batch in it:
            base_inputs = batch_collate(batch) if batch_collate is not None else batch
            if isinstance(batch, (list, tuple)) and len(batch) == 1:
                base_inputs = batch[0]

            # Determine batch size before running the model
            def _batch_size_of(obj) -> int:
                if isinstance(obj, torch.Tensor):
                    return int(obj.shape[0])
                if isinstance(obj, Mapping):
                    for v in obj.values():
                        if isinstance(v, torch.Tensor):
                            return int(v.shape[0])
                if isinstance(obj, (list, tuple)) and len(obj) > 0:
                    if isinstance(obj[0], torch.Tensor):
                        return int(obj[0].shape[0])
                return 0

            n_in_batch = _batch_size_of(base_inputs)
            if rows_to_skip >= n_in_batch and n_in_batch > 0:
                rows_to_skip -= n_in_batch
                continue

            base_inputs = self._move_to_device(base_inputs, self.base_device)
            # Mixed precision context to reduce memory if requested
            if amp_dtype is not None and str(self.base_device).startswith("cuda"):
                amp_map = {"bf16": torch.bfloat16, "float16": torch.float16, "fp16": torch.float16}
                amp_t = amp_map.get(str(amp_dtype).lower())
                ctx = torch.cuda.amp.autocast(dtype=amp_t) if amp_t is not None else contextlib.nullcontext()
            else:
                ctx = contextlib.nullcontext()
            with ctx:
                if isinstance(base_inputs, Mapping):
                    _ = self.base_model(**base_inputs)
                else:
                    _ = self.base_model(base_inputs)
            feats = self._pop_cached_activation()

            # If partially skipping within this batch, slice post-compute
            if rows_to_skip > 0:
                feats = feats[rows_to_skip:]
                rows_to_skip = 0
            if feat_dim is None:
                feat_dim = int(feats.shape[-1])
            shard_rows.append(feats.detach().cpu())
            shard_row_count += feats.shape[0]
            total += feats.shape[0]

            if shard_row_count >= shard_size:
                tensor = torch.cat(shard_rows, dim=0).to(store_dtype)
                path = os.path.join(out_dir, f"feats_{shard_idx:05d}.pt")
                torch.save(tensor, path)
                shards_meta.append({"file": os.path.basename(path), "count": int(tensor.shape[0])})
                shard_rows.clear()
                shard_row_count = 0
                shard_idx += 1
                if show_progress:
                    tqdm.write(f"Saved shard {shard_idx} to {path} [{tensor.shape[0]} rows]")

        if shard_row_count > 0:
            tensor = torch.cat(shard_rows, dim=0).to(store_dtype)
            path = os.path.join(out_dir, f"feats_{shard_idx:05d}.pt")
            torch.save(tensor, path)
            shards_meta.append({"file": os.path.basename(path), "count": int(tensor.shape[0])})
            if show_progress:
                tqdm.write(f"Saved shard {shard_idx} to {path} [{tensor.shape[0]} rows]")

        manifest = {
            "total": int((resume_rows if resume_manifest else 0) + total),
            "dim": int(feat_dim or 0),
            "dtype": dtype,
            "shard_size": int(shard_size),
            "shards": shards_meta,
        }
        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        return manifest

    # ----------------------------------- train ----------------------------------
    def fit(
        self,
        X_train,
        X_val: Optional[object] = None,
        *,
        batch_size: int = 64,
        learning_rate: float = 5e-4,
        n_epochs: int = 10,
        patience: int = 3,
        clip_grad: Optional[float] = 1.0,
        show_progress: bool = True,
        cache_embeddings: bool = True,
    ) -> Dict[str, list]:
        # Build flexible datasets that can handle tensors, dicts of tensors, or lists of items
        from torch.utils.data import Dataset

        class ItemsDataset(Dataset):  # type: ignore[type-arg]
            def __init__(self, data):
                self.data = data

            def __len__(self):
                d = self.data
                if isinstance(d, torch.Tensor):
                    return d.shape[0]
                if isinstance(d, Mapping):
                    for v in d.values():
                        if isinstance(v, torch.Tensor):
                            return v.shape[0]
                    raise TypeError("Dict values must include at least one tensor to infer length")
                if isinstance(d, (list, tuple)):
                    return len(d)
                raise TypeError("Unsupported dataset type for MountedSAE.fit")

            def __getitem__(self, idx):
                d = self.data
                if isinstance(d, torch.Tensor):
                    return d[idx]
                if isinstance(d, Mapping):
                    return {k: (v[idx] if isinstance(v, torch.Tensor) else v) for k, v in d.items()}
                if isinstance(d, (list, tuple)):
                    return d[idx]
                raise TypeError("Unsupported dataset type for MountedSAE.fit")

        # Build raw loaders over inputs; may be replaced by cached feature loaders below
        if isinstance(X_train, torch.Tensor):
            raw_train_loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=False)
        else:
            raw_train_loader = DataLoader(ItemsDataset(X_train), batch_size=batch_size, shuffle=False)

        if X_val is None:
            raw_val_loader = None
        elif isinstance(X_val, torch.Tensor):
            raw_val_loader = DataLoader(TensorDataset(X_val), batch_size=batch_size)
        else:
            raw_val_loader = DataLoader(ItemsDataset(X_val), batch_size=batch_size)

        # Optionally cache features to avoid recomputing base model activations every epoch
        def _compute_features(data_loader, desc: str) -> torch.Tensor:
            feats_list = []
            it = tqdm(data_loader, desc=desc, leave=False) if show_progress else data_loader
            with torch.no_grad():
                for batch in it:
                    base_inputs = batch
                    if isinstance(batch, (list, tuple)) and len(batch) == 1:
                        base_inputs = batch[0]
                    base_inputs = self._move_to_device(base_inputs, self.base_device)
                    if isinstance(base_inputs, Mapping):
                        _ = self.base_model(**base_inputs)
                    else:
                        _ = self.base_model(base_inputs)
                    feats = self._pop_cached_activation()
                    feats_list.append(feats.detach().cpu())
            return torch.cat(feats_list, dim=0) if feats_list else torch.zeros(0)

        if cache_embeddings:
            train_feats = _compute_features(raw_train_loader, desc="Cache train feats")
            train_loader = DataLoader(TensorDataset(train_feats), batch_size=batch_size, shuffle=True)
            if raw_val_loader is not None:
                val_feats = _compute_features(raw_val_loader, desc="Cache val feats")
                val_loader = DataLoader(TensorDataset(val_feats), batch_size=batch_size, shuffle=False)
            else:
                val_loader = None
            using_cached = True
        else:
            train_loader = raw_train_loader
            val_loader = raw_val_loader
            using_cached = False

        # Optimizer only on SAE parameters
        # SAE may be lazily created on the first batch; create a temporary optimizer and rebuild if needed
        opt: Optional[torch.optim.Optimizer] = torch.optim.Adam(self.sae.parameters(), lr=learning_rate) if self.sae is not None else None  # type: ignore[arg-type]

        best_val = float("inf")
        patience_ctr = 0
        history = {"train_loss": [], "val_loss": []}

        epoch_iter = tqdm(range(n_epochs), desc="Epochs") if show_progress else range(n_epochs)
        for _ in epoch_iter:
            # ----------------------------- training epoch -----------------------------
            train_losses = []
            if self.sae is not None:
                self.sae.train()
            train_batch_iter = tqdm(train_loader, desc="Train", leave=False) if show_progress else train_loader
            for batch in train_batch_iter:
                # Obtain features either from cache or by running the base model
                if using_cached:
                    feats = batch[0] if isinstance(batch, (list, tuple)) else batch
                else:
                    with torch.no_grad():
                        base_inputs = batch
                        if isinstance(batch, (list, tuple)) and len(batch) == 1:
                            base_inputs = batch[0]
                        base_inputs = self._move_to_device(base_inputs, self.base_device)
                        if isinstance(base_inputs, Mapping):
                            _ = self.base_model(**base_inputs)
                        else:
                            _ = self.base_model(base_inputs)
                    feats = self._pop_cached_activation()
                # Initialize SAE lazily if needed
                if self.sae is None:
                    self._ensure_sae_initialized(feats)
                    opt = torch.optim.Adam(self.sae.parameters(), lr=learning_rate)  # type: ignore[arg-type]
                    self.sae.train()  # type: ignore[union-attr]

                x_hat, info = self.sae(feats.to(self.sae.device))  # type: ignore[union-attr]
                loss = criterion(
                    feats.to(self.sae.device),  # type: ignore[union-attr]
                    x_hat,
                    info["pre_codes"],
                    info["codes"],
                    info["dictionary"],
                )

                opt.zero_grad()  # type: ignore[union-attr]
                loss.backward()
                self.sae.adjust_decoder_gradient_()  # type: ignore[union-attr]
                if clip_grad is not None:
                    nn.utils.clip_grad_norm_(self.sae.parameters(), clip_grad)  # type: ignore[union-attr]
                opt.step()  # type: ignore[union-attr]
                self.sae.normalize_decoder_()  # type: ignore[union-attr]
                train_losses.append(loss.item())

            avg_train = float(sum(train_losses) / max(1, len(train_losses)))
            history["train_loss"].append(avg_train)

            # ----------------------------- validation epoch ----------------------------
            if val_loader is not None:
                self.sae.eval()  # type: ignore[union-attr]
                val_losses = []
                with torch.no_grad():
                    val_batch_iter = tqdm(val_loader, desc="Val", leave=False) if show_progress else val_loader
                    for batch in val_batch_iter:
                        if using_cached:
                            feats = batch[0] if isinstance(batch, (list, tuple)) else batch
                        else:
                            base_inputs = batch
                            if isinstance(batch, (list, tuple)) and len(batch) == 1:
                                base_inputs = batch[0]
                            base_inputs = self._move_to_device(base_inputs, self.base_device)
                            if isinstance(base_inputs, Mapping):
                                _ = self.base_model(**base_inputs)
                            else:
                                _ = self.base_model(base_inputs)
                            feats = self._pop_cached_activation()
                        x_hat, info = self.sae(feats.to(self.sae.device))  # type: ignore[union-attr]
                        vloss = criterion(
                            feats.to(self.sae.device),  # type: ignore[union-attr]
                            x_hat,
                            info["pre_codes"],
                            info["codes"],
                            info["dictionary"],
                        )
                        val_losses.append(vloss.item())
                avg_val = float(sum(val_losses) / max(1, len(val_losses)))
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
                postfix = {"train": f"{avg_train:.4f}"}
                if val_loader is not None:
                    postfix["val"] = f"{history['val_loss'][-1]:.4f}"
                epoch_iter.set_postfix(postfix)

        return history

    # ---------------------------- batched SAE inference ---------------------------
    @torch.no_grad()
    def get_activations(self, inputs, batch_size: int = 256, show_progress: bool = True) -> torch.Tensor:
        self.sae.eval()  # type: ignore[union-attr]
        total = inputs.shape[0]
        codes = []
        rng = tqdm(range(0, total, batch_size), desc=f"MountedSAE Activations (batch={batch_size})") if show_progress else range(0, total, batch_size)
        for i in rng:
            batch = inputs[i : i + batch_size]
            _ = self.base_model(batch.to(self.base_device))
            feats = self._pop_cached_activation()
            _, info = self.sae(feats.to(self.sae.device))  # type: ignore[union-attr]
            codes.append(info["codes"].cpu())
        return torch.cat(codes, dim=0)


