#!/usr/bin/env python3
import argparse
import json
import hashlib
import os
import time
from typing import List, Tuple, Optional, Sequence
import logging

import torch

# Package imports
from saerm.rm.btrm import BTRM
from saerm.process_data.litbench_hf import LitBenchHF
from saerm.sae.mounted import MountedSAE
from saerm.eval.sae_metrics import compute_sae_metrics
from saerm.data.feature_shards import FeatureShardBatches


def _repo_subdir(repo: str) -> str:
    """Return a nested subdirectory path mirroring repo id, e.g. owner/model."""
    parts = [p for p in str(repo).split("/") if p]
    return os.path.join(*parts) if parts else ""


def _select_last_token(hidden_states: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # Select the last non-pad token embedding per sequence
    idx = attention_mask.sum(dim=1) - 1
    idx = torch.clamp(idx, min=0)
    b = hidden_states.size(0)
    return hidden_states[torch.arange(b, device=hidden_states.device), idx, :]


def _activation_transform_from_output(output):
    # Transform a SequenceClassifierOutput with hidden_states into a 2D tensor (B, D)
    hidden_states = getattr(output, "hidden_states", None)
    if hidden_states is None:
        raise RuntimeError("Model output does not contain hidden_states. Ensure config.output_hidden_states=True.")
    # Expect input_ids and attention_mask present in cached inputs via hook context; not available here
    # Instead, rely on last hidden state and simple last-token selection using attention_mask from model inputs
    # MountedSAE's hook only passes 'output', so we attach input_ids/attention_mask to the model for retrieval
    model = output.__dict__.get("_saerm_attached_model", None)
    if model is None or not hasattr(model, "_saerm_last_batch"):
        # Fallback: pool by mean over tokens
        last_h = hidden_states[-1]
        return last_h.mean(dim=1)
    batch = model._saerm_last_batch
    last_h = hidden_states[-1]
    return _select_last_token(last_h, batch["input_ids"], batch["attention_mask"]).float()


def main():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Train a MountedSAE on the last layer of a loaded BTRM model (YAML-configured).")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    try:
        import yaml  # type: ignore
    except Exception as e:
        raise SystemExit("PyYAML is required. Install with: pip install pyyaml") from e

    logging.info("Loading config: %s", args.config)
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Required sections and keys
    btrm_cfg = cfg.get("btrm", {})
    sae_cfg = cfg.get("sae", {})
    out_cfg = cfg.get("outputs", {})
    wandb_cfg = cfg.get("wandb", {})

    repo = btrm_cfg["repo"]
    # Load .env if present and prefer env var HF_TOKEN
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except Exception:
        pass
    # Silence TRANSFORMERS_CACHE deprecation by mapping to HF_HOME early
    if os.environ.get("TRANSFORMERS_CACHE") and not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = os.environ["TRANSFORMERS_CACHE"]
        del os.environ["TRANSFORMERS_CACHE"]
        logging.info("Using HF_HOME at %s (migrated from TRANSFORMERS_CACHE)", os.environ.get("HF_HOME"))
    hf_token = os.environ.get("HF_TOKEN", "") or btrm_cfg.get("hf_token") or ""
    device = btrm_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    max_length = int(btrm_cfg["max_length"])

    num_neurons = int(sae_cfg["num_neurons"])
    k_active = int(sae_cfg["k_active"])
    train_cfg = sae_cfg.get("training", {})
    batch_size = int(train_cfg["batch_size"])
    learning_rate = float(train_cfg["learning_rate"])
    n_epochs = int(train_cfg["n_epochs"])
    patience = int(train_cfg["patience"])
    show_progress = bool(train_cfg.get("show_progress", True))

    outputs_dir = out_cfg["dir"]

    os.makedirs(outputs_dir, exist_ok=True)

    # --------------------- Load BTRM (HF model + tokenizer) ---------------------
    logging.info("Loading base model and tokenizer: %s", repo)
    btrm = BTRM.load(
        repo,
        {
            "hf_token": hf_token,
            "max_length": max_length,
            "device": device,
            "trust_remote_code": False,
        },
    )

    # Enable hidden states and attach a tiny context to retrieve input batch inside hook
    btrm._model.config.output_hidden_states = True  # type: ignore[attr-defined]

    # Wrap the model forward to capture last tokenizer batch for selection logic
    orig_forward = btrm._model.forward  # type: ignore[attr-defined]

    def forward_with_batch_capture(*fw_args, **fw_kwargs):  # type: ignore[no-redef]
        out = orig_forward(*fw_args, **fw_kwargs)
        # Attach the last batch tensors for the hook-side transform
        batch_like = {}
        for k, v in fw_kwargs.items():
            if isinstance(v, torch.Tensor):
                batch_like[k] = v
        # If tokenized dict was passed as a single kwarg (rare), also support that
        for v in fw_kwargs.values():
            if isinstance(v, dict) and all(isinstance(t, torch.Tensor) for t in v.values()):
                batch_like.update(v)
        setattr(btrm._model, "_saerm_last_batch", batch_like)  # type: ignore[attr-defined]
        # Also attach back-reference for activation transform
        setattr(out, "_saerm_attached_model", btrm._model)  # type: ignore[attr-defined]
        return out

    btrm._model.forward = forward_with_batch_capture  # type: ignore[attr-defined]

    # ----------------------------- Load LitBench -----------------------------
    logging.info("Loading dataset (LitBenchHF)")
    ds = LitBenchHF(hf_token=hf_token)
    train_pairs = ds.train()

    # Build training/validation texts from chosen+rejected
    def pairs_to_texts(pairs: List) -> List[str]:
        out: List[str] = []
        for p in pairs:
            out.append(p.chosen)
            out.append(p.rejected)
        return out

    texts_all = pairs_to_texts(train_pairs)
    logging.info("Prepared %d training texts", len(texts_all))

    tok = btrm._tokenizer

    def encode(texts: List[str]) -> dict:
        return tok(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

    # Train on all data; no separate validation
    # Two-phase controls
    features_cfg = cfg.get("features", {})
    features_dir_root: Optional[str] = features_cfg.get("dir")
    shard_size = int(features_cfg.get("shard_size", 20000))
    feats_dtype = str(features_cfg.get("dtype", "float16"))
    extract_only = bool(features_cfg.get("extract_only", False))
    train_from_features = bool(features_cfg.get("train_from_features", bool(features_dir_root)))
    resume_before_train = bool(features_cfg.get("resume_before_train", True))

    # Resolve per-model features directory if a root is provided
    features_dir: Optional[str] = None
    if isinstance(features_dir_root, str) and features_dir_root.strip() != "":
        repo_sub = _repo_subdir(repo)
        features_dir = os.path.join(features_dir_root, repo_sub)
        try:
            os.makedirs(features_dir, exist_ok=True)
        except Exception:
            pass

    # Encoder collate for streaming tokenization during extraction
    def _collate_text_batch(batch: Sequence[object]) -> dict:
        if not isinstance(batch, (list, tuple)):
            return batch  # already dict/tensor
        # batch is a list of either strings or dict/tuple; support str list here
        items: List[str] = []
        for it in batch:
            if isinstance(it, str):
                items.append(it)
            elif isinstance(it, dict):
                # already tokenized
                return it
            else:
                raise TypeError("Unexpected batch item type for text extraction")
        return encode(items)

    # If training from features, optionally ensure features are complete by resuming extraction first
    if train_from_features and features_dir and resume_before_train:
        logging.info("Resuming/ensuring features complete in %s (shard_size=%d, dtype=%s)", features_dir, shard_size, feats_dtype)
        # Build a fresh MountedSAE for extraction
        device_model = btrm._model.device  # type: ignore[attr-defined]
        mounted_for_extract = MountedSAE(
            base_model=btrm._model,
            layer=btrm._model,
            num_neurons=num_neurons,
            k_active=k_active,
            device=str(device_model),
            flatten=False,
            activation_transform=_activation_transform_from_output,
        )
        mounted_for_extract.extract_features_to_dir(texts_all, features_dir, batch_size=batch_size, shard_size=shard_size, dtype=feats_dtype, show_progress=True, batch_collate=_collate_text_batch)

    if train_from_features and features_dir:
        logging.info("Training SAE from precomputed features in %s", features_dir)
        stream = FeatureShardBatches(features_dir, batch_size=batch_size)
        input_dim = stream.dim
        # -------------------------- Deduplicate features --------------------------
        # Build a boolean mask per shard indicating unique rows across all shards
        # We compute a rolling SHA1 over rows to avoid storing all rows in memory
        logging.info("Scanning feature shards for duplicates (streaming hash)...")
        with open(os.path.join(features_dir, "manifest.json"), "r") as f:
            manifest = json.load(f)
        shards = manifest["shards"]
        unique_hashes = set()
        shard_unique_masks = []
        total_rows = 0
        unique_rows = 0
        for sh in shards:
            path = os.path.join(features_dir, sh["file"])  # type: ignore[index]
            t = torch.load(path, map_location="cpu")
            t = t.float()
            # Compute per-row hashes
            m = []
            for row in t:
                b = row.numpy().tobytes()
                h = hashlib.sha1(b).hexdigest()
                if h in unique_hashes:
                    m.append(False)
                else:
                    unique_hashes.add(h)
                    m.append(True)
            mask = torch.tensor(m, dtype=torch.bool)
            shard_unique_masks.append({"file": sh["file"], "mask": mask})
            total_rows += int(t.shape[0])
            unique_rows += int(mask.sum().item())
        logging.info("Feature dedup: total=%d, unique=%d, removed=%d", total_rows, unique_rows, total_rows - unique_rows)
        print(json.dumps({"features_total": total_rows, "features_unique": unique_rows}))
        mounted = None  # not needed for training-from-features
        # Build SAE and train using stream
        from saerm.sae.batchtopk_sae import BatchTopKSAE, criterion
        sae = BatchTopKSAE(input_dim=input_dim, num_neurons=num_neurons, k_active=k_active, device=str(device))
        opt = torch.optim.Adam(sae.parameters(), lr=learning_rate)
        from tqdm.auto import tqdm
        # Optional Weights & Biases logging
        use_wandb = bool(wandb_cfg.get("enabled", False))
        wandb_run = None
        if use_wandb:
            try:
                import wandb  # type: ignore
                wandb_run = wandb.init(
                    project=str(wandb_cfg.get("project", "saerm")),
                    name=str(wandb_cfg.get("run_name", os.path.basename(outputs_dir) + "-sae")),
                    config={
                        "repo": repo,
                        "device": str(device),
                        "max_length": max_length,
                        "num_neurons": num_neurons,
                        "k_active": k_active,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "n_epochs": n_epochs,
                    },
                )
            except Exception as _wandb_err:
                logging.warning("wandb init failed; continuing without logging: %s", _wandb_err)
                use_wandb = False
        best = float("inf")
        bad = 0
        for epoch_idx in tqdm(range(n_epochs), desc="Epochs"):
            sae.train()
            losses = []
            # Accumulators for metrics
            mse_sum = 0.0
            mse_batches = 0
            sse = 0.0
            sum_all = 0.0
            sumsq_all = 0.0
            total_count = 0
            sparsity_sum = 0.0
            sparsity_batches = 0
            # Re-iterate shards and apply masks to get only unique rows for training
            for mask_info in shard_unique_masks:
                path = os.path.join(features_dir, mask_info["file"])  # type: ignore[index]
                t = torch.load(path, map_location="cpu").float()
                mask = mask_info["mask"]
                if mask.numel() != t.shape[0]:
                    # If shard size changed, fall back to training on all rows of this shard
                    sel = t
                else:
                    sel = t[mask]
                if sel.numel() == 0:
                    continue
                feats = sel.to(device)
                x_hat, info = sae(feats)
                loss = criterion(feats, x_hat, info["pre_codes"], info["codes"], info["dictionary"])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                sae.adjust_decoder_gradient_()
                torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
                opt.step()
                sae.normalize_decoder_()
                losses.append(loss.item())
                # Metrics per batch
                with torch.no_grad():
                    diff = feats - x_hat
                    mse_batch = diff.pow(2).mean().item()
                    mse_sum += float(mse_batch)
                    mse_batches += 1
                    sse += float(diff.pow(2).sum().item())
                    sum_all += float(feats.sum().item())
                    sumsq_all += float(feats.pow(2).sum().item())
                    total_count += int(feats.numel())
                    density = (info["codes"] > 0).float().mean().item()
                    sparsity_sum += float(1.0 - density)
                    sparsity_batches += 1
            avg = float(sum(losses) / max(1, len(losses)))
            avg_mse = float(mse_sum / max(1, mse_batches))
            sst = float(sumsq_all - (total_count * (sum_all / max(1.0, float(total_count))) ** 2)) if total_count > 0 else 0.0
            r2 = float(1.0 - (sse / sst)) if sst > 0 else 0.0
            avg_sparsity = float(sparsity_sum / max(1, sparsity_batches))
            logging.info(
                "Epoch %d/%d - loss=%.6f, mse=%.6f, r2=%.4f, sparsity=%.4f",
                int(epoch_idx) + 1,
                int(n_epochs),
                avg,
                avg_mse,
                r2,
                avg_sparsity,
            )
            if avg < best - 1e-8:
                best = avg
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break
            if use_wandb:
                try:
                    import wandb  # type: ignore
                    wandb.log({
                        "epoch": int(epoch_idx),
                        "train/loss": avg,
                        "train/mse": avg_mse,
                        "train/r2": r2,
                        "train/sparsity": avg_sparsity,
                    })
                except Exception as _wandb_log_err:
                    logging.warning("wandb.log failed: %s", _wandb_log_err)

        # Save outputs consistent with previous script
        os.makedirs(os.path.join(outputs_dir, "sae"), exist_ok=True)
        torch.save({
            "state_dict": sae.state_dict(),
            "num_neurons": num_neurons,
            "k_active": k_active,
        }, os.path.join(outputs_dir, "sae", "sae.pt"))
        logging.info("Saved SAE to %s", os.path.join(outputs_dir, "sae", "sae.pt"))
        if use_wandb and wandb_run is not None:
            try:
                wandb_run.finish()  # type: ignore[attr-defined]
            except Exception:
                pass
        print(json.dumps({"outputs": outputs_dir}))
        return

    # Default: run extraction (optional), then standard train from raw inputs
    X_train = texts_all  # keep texts; collate will tokenize per-batch
    X_val = None

    device = btrm._model.device  # type: ignore[attr-defined]
    # ------------------------- Mount and train the SAE -------------------------
    mounted = MountedSAE(
        base_model=btrm._model,
        layer=btrm._model,  # hook top-level; transform extracts last hidden states
        num_neurons=num_neurons,
        k_active=k_active,
        device=str(device),
        flatten=False,
        activation_transform=_activation_transform_from_output,
    )

    # Phase 1: optional extraction to features_dir
    if features_dir:
        logging.info("Extracting features to %s (shard_size=%d, dtype=%s)", features_dir, shard_size, feats_dtype)
        # Pass raw texts; extractor will build its own loader and collate/tokenize per-batch
        mounted.extract_features_to_dir(texts_all, features_dir, batch_size=batch_size, shard_size=shard_size, dtype=feats_dtype, show_progress=True, batch_collate=_collate_text_batch)
        if extract_only:
            logging.info("Extraction complete. Exiting because extract_only=true")
            return

    # Phase 2: standard in-process training from raw inputs (no features)
    logging.info("Starting in-process training (no external features)")
    hist = mounted.fit(
        encode(texts_all),
        X_val=X_val,
        batch_size=batch_size,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        patience=patience,
        show_progress=show_progress,
    )

    # ----------------------- Report sparsity and R^2 -----------------------
    eval_cap = int(out_cfg.get("metrics_cap", 2048))
    use_texts = texts_all
    enc_inputs = encode(use_texts[:eval_cap])
    enc_inputs = {k: v.to(device) for k, v in enc_inputs.items()}
    sparsity, r2 = compute_sae_metrics(mounted, enc_inputs, sample_cap=None)

    report = {
        "timestamp": int(time.time()),
        "device": str(device),
        "train_pairs_used": len(texts_all) // 2,
        "val_pairs_used": 0,
        "history": hist,
        "sparsity": sparsity,
        "r2": r2,
        "sae": {
            "num_neurons": num_neurons,
            "k_active": k_active,
        },
        "btrm": {
            "repo": repo,
            "max_length": max_length,
        },
    }

    with open(os.path.join(outputs_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # Save SAE weights and config
    sae_dir = os.path.join(outputs_dir, "sae")
    os.makedirs(sae_dir, exist_ok=True)
    torch.save(
        {
            "state_dict": mounted.sae.state_dict() if mounted.sae is not None else None,
            "num_neurons": num_neurons,
            "k_active": k_active,
        },
        os.path.join(sae_dir, "sae.pt"),
    )

    print(json.dumps({"sparsity": sparsity, "r2": r2, "outputs": outputs_dir}, indent=2))


if __name__ == "__main__":
    main()


