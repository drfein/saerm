#!/usr/bin/env python3
import argparse
import json
import os
import time
import logging

import torch

from saerm.rm.btrm import BTRM
from saerm.sae.mounted import MountedSAE
from saerm.rm.saerm import SAERM
from saerm.process_data.litbench_hf import LitBenchHF
from saerm.eval.preference import PreferenceEvaluator


def _repo_subdir(repo: str) -> str:
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
    # MountedSAE's hook passes only 'output', so retrieve batch via attached context
    model = output.__dict__.get("_saerm_attached_model", None)
    if model is None or not hasattr(model, "_saerm_last_batch"):
        # Fallback: mean over tokens
        last_h = hidden_states[-1]
        return last_h.mean(dim=1)
    batch = model._saerm_last_batch
    last_h = hidden_states[-1]
    return _select_last_token(last_h, batch["input_ids"], batch["attention_mask"]).float()


def main():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Train a SAERM head on a saved SAE's activations and report test accuracy (YAML-configured).")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    try:
        import yaml  # type: ignore
    except Exception as e:
        raise SystemExit("PyYAML is required. Install with: pip install pyyaml") from e

    logging.info("Loading config: %s", args.config)
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    btrm_cfg = cfg.get("btrm", {})
    sae_cfg = cfg.get("sae", {})
    rm_cfg = cfg.get("rm", {})
    out_cfg = cfg.get("outputs", {})

    # .env support for HF_TOKEN
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except Exception:
        pass

    sae_dir = sae_cfg["dir"]
    sae_features_root = sae_cfg.get("features_dir")
    repo = btrm_cfg["repo"]
    hf_token = os.environ.get("HF_TOKEN", "") or btrm_cfg.get("hf_token") or ""
    device = btrm_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    max_length = int(btrm_cfg["max_length"])
    encode_batch_size = int(rm_cfg.get("encode_batch_size", 32))
    outputs_dir = out_cfg["dir"]
    show_progress = bool(rm_cfg.get("show_progress", True))

    # Optional eval feature cache root (namespace per model repo)
    eval_features_root = rm_cfg.get("eval_features_dir")

    os.makedirs(outputs_dir, exist_ok=True)

    # Resolve per-model features directories if provided
    sae_features_dir = None
    eval_features_dir = None
    repo_sub = _repo_subdir(repo)
    if isinstance(sae_features_root, str) and sae_features_root.strip() != "":
        sae_features_dir = os.path.join(sae_features_root, repo_sub)
    if isinstance(eval_features_root, str) and eval_features_root.strip() != "":
        eval_features_dir = os.path.join(eval_features_root, repo_sub)

    # -------------------------- Load BTRM and SAE --------------------------
    logging.info("Loading BTRM model and tokenizer: %s", repo)
    btrm = BTRM.load(
        repo,
        {
            "hf_token": hf_token,
            "max_length": max_length,
            "device": device,
            "trust_remote_code": False,
        },
    )
    btrm._model.config.output_hidden_states = True  # type: ignore[attr-defined]

    # Attach batch-capture wrapper so activation_transform can read true attention_mask
    orig_forward = btrm._model.forward  # type: ignore[attr-defined]

    def forward_with_batch_capture(*fw_args, **fw_kwargs):  # type: ignore[no-redef]
        out = orig_forward(*fw_args, **fw_kwargs)
        # Collect tensor kwargs (and nested dict) for token selection
        batch_like = {}
        for k, v in fw_kwargs.items():
            if isinstance(v, torch.Tensor):
                batch_like[k] = v
        for v in fw_kwargs.values():
            if isinstance(v, dict) and all(isinstance(t, torch.Tensor) for t in v.values()):
                batch_like.update(v)
        setattr(btrm._model, "_saerm_last_batch", batch_like)  # type: ignore[attr-defined]
        setattr(out, "_saerm_attached_model", btrm._model)  # type: ignore[attr-defined]
        return out

    btrm._model.forward = forward_with_batch_capture  # type: ignore[attr-defined]

    # Recreate MountedSAE and load SAE weights
    logging.info("Loading SAE from %s", sae_dir)
    sae_ckpt_path = os.path.join(sae_dir, "sae.pt")
    ckpt = torch.load(sae_ckpt_path, map_location=btrm._model.device)  # type: ignore[attr-defined]
    num_neurons = int(ckpt.get("num_neurons"))
    k_active = int(ckpt.get("k_active"))

    mounted = MountedSAE(
        base_model=btrm._model,
        layer=btrm._model,  # top-level; uses last hidden states implicitly
        num_neurons=num_neurons,
        k_active=k_active,
        device=str(btrm._model.device),  # type: ignore[attr-defined]
        flatten=False,
        activation_transform=_activation_transform_from_output,
    )
    # Lazily initialize SAE by a tiny forward pass to get input_dim, then load weights
    logging.info("Initializing SAE (dummy forward to infer input_dim)")
    with torch.no_grad():
        dummy = btrm._tokenizer(["hello"], return_tensors="pt", padding=True).to(btrm._model.device)
        _ = btrm._model(**dummy)
        feats = mounted._pop_cached_activation()
        mounted._ensure_sae_initialized(feats)
    if mounted.sae is None:
        raise RuntimeError("Failed to initialize SAE before loading weights")
    mounted.sae.load_state_dict(ckpt["state_dict"])  # type: ignore[index]
    mounted.sae.eval()
    logging.info("SAE ready: input_dim=%d, neurons=%d, k_active=%d", feats.shape[-1], num_neurons, k_active)

    # ------------------------ Build encode function ------------------------
    tok = btrm._tokenizer
    device = btrm._model.device  # type: ignore[attr-defined]

    def encode_fn(texts):
        return tok(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)

    # SAERM that uses mounted SAE codes as features with linear BT head
    saerm = SAERM(sae=mounted, encode_fn=encode_fn, device=str(device), encode_batch_size=encode_batch_size)

    # ------------------------------ Data & Train ------------------------------
    logging.info("Loading preference dataset (LitBenchHF)")
    ds = LitBenchHF(hf_token=hf_token)
    # If precomputed features are available, train SAERM head from them to avoid re-encoding
    if sae_features_dir and os.path.exists(os.path.join(sae_features_dir, "manifest.json")):
        logging.info("Found precomputed features in %s; training SAERM head from features", sae_features_dir)
        with open(os.path.join(sae_features_dir, "manifest.json"), "r") as f:
            manifest = json.load(f)
        shards = manifest.get("shards", [])
        total_rows = int(manifest.get("total", 0))
        if total_rows % 2 != 0:
            logging.warning("Feature total rows (%d) is not even; chosen/rejected pairing may be off", total_rows)

        # Build codes for chosen (even rows) and rejected (odd rows)
        Z_ch_list = []
        Z_rj_list = []
        sae_inst = mounted.sae
        if sae_inst is None:
            raise RuntimeError("SAE instance missing after load")
        sae_device = sae_inst.device  # type: ignore[attr-defined]
        try:
            from tqdm.auto import tqdm  # type: ignore
            shard_iter = tqdm(shards, desc="Codes from features", leave=False)
        except Exception:
            shard_iter = shards

        row_cursor = 0
        for sh in shard_iter:
            path = os.path.join(sae_features_dir, sh["file"])  # type: ignore[index]
            feats_cpu = torch.load(path, map_location="cpu").float()
            # Process in chunks to manage memory
            step = max(1, encode_batch_size)
            for i in range(0, feats_cpu.shape[0], step):
                chunk = feats_cpu[i : i + step].to(sae_device)
                with torch.no_grad():
                    _xhat, info = sae_inst(chunk)
                    codes = info["codes"].detach().to("cpu")
                # Split by global index parity
                b = codes.shape[0]
                idxs = torch.arange(row_cursor, row_cursor + b)
                even_mask = (idxs % 2 == 0)
                odd_mask = ~even_mask
                if even_mask.any():
                    Z_ch_list.append(codes[even_mask])
                if odd_mask.any():
                    Z_rj_list.append(codes[odd_mask])
                row_cursor += b

        if len(Z_ch_list) == 0 or len(Z_rj_list) == 0:
            raise RuntimeError("No codes extracted from features; cannot train SAERM head")
        Z_ch = torch.cat(Z_ch_list, dim=0).to(device)
        Z_rj = torch.cat(Z_rj_list, dim=0).to(device)
        n_pairs = min(Z_ch.shape[0], Z_rj.shape[0])
        if Z_ch.shape[0] != Z_rj.shape[0]:
            logging.warning("Mismatched chosen/rejected counts: %d vs %d; truncating to %d", Z_ch.shape[0], Z_rj.shape[0], n_pairs)
            Z_ch = Z_ch[:n_pairs]
            Z_rj = Z_rj[:n_pairs]

        # Standardize and train linear BT head (mirrors SAERM.train)
        mu = Z_ch.mean(dim=0)
        sig = Z_ch.std(dim=0).clamp_min(1e-8)
        Z_ch_std = (Z_ch - mu) / sig
        Z_rj_std = (Z_rj - mu) / sig

        m = Z_ch_std.shape[1]
        w = torch.zeros(m, device=device, requires_grad=True)
        lr = 1e-2
        wd = 1e-4
        epochs = int(rm_cfg.get("bt_epochs", 10))
        logging.info("BT head training epochs: %d", epochs)
        batch_size_bt = 4096
        opt = torch.optim.AdamW([w], lr=lr, weight_decay=wd)
        try:
            from tqdm.auto import tqdm  # type: ignore
            epoch_iter = tqdm(range(epochs), desc="BT head epochs")
        except Exception:
            epoch_iter = range(epochs)
        n = Z_ch_std.size(0)
        for _ in epoch_iter:
            perm = torch.randperm(n, device=device)
            loss_sum = 0.0
            batch_ctr = 0
            correct = 0.0
            total_pairs = 0
            for i in range(0, n, batch_size_bt):
                idx = perm[i : i + batch_size_bt]
                zc = Z_ch_std[idx]
                zr = Z_rj_std[idx]
                s_ch = zc @ w
                s_rj = zr @ w
                diff = s_ch - s_rj
                loss = torch.nn.functional.softplus(-diff).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                # metrics accumulators
                loss_sum += float(loss.item())
                batch_ctr += 1
                correct += float((diff > 0).float().mean().item())
                total_pairs += 1
            avg_loss = loss_sum / max(1, batch_ctr)
            acc = correct / max(1, total_pairs)
            try:
                # Update tqdm postfix if available
                epoch_iter.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{acc:.3f}"})  # type: ignore[attr-defined]
            except Exception:
                pass
            logging.info("BT epoch - loss=%.6f acc=%.3f", avg_loss, acc)

        # Install trained head and stats into SAERM instance
        saerm._w = w.detach()
        saerm._mu = mu.detach()
        saerm._sig = sig.detach()
        logging.info("SAERM head trained from features: pairs=%d, dim=%d", int(n_pairs), int(m))
    else:
        # Fallback: encode texts via base model (slower)
        # Add simple tqdm for SAERM training epochs/batches by monkey-patching when possible
        try:
            from tqdm.auto import tqdm  # type: ignore
            logging.info("Training SAERM head with BT loss")
            # Since SAERM.train encodes all pairs then runs fixed epochs internally, we can't inject tqdm easily
            # but we can at least time it and inform start/end.
        except Exception:
            logging.info("Training SAERM head (tqdm unavailable)")
        start_t = time.time()
        saerm.train(ds)
        logging.info("SAERM training complete in %.2fs", time.time() - start_t)

    # Save head and stats
    logging.info("Saving SAERM head and stats to %s", outputs_dir)
    saerm.save(outputs_dir)

    # ------------------------- Optional: Save eval base features -------------------------
    if isinstance(eval_features_dir, str):
        manifest_path = os.path.join(eval_features_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            try:
                os.makedirs(eval_features_dir, exist_ok=True)
            except Exception:
                pass
            logging.info("Extracting and saving eval base-model features to %s", eval_features_dir)
            # Reuse the mounted SAE extractor to dump base features; it stores layer outputs captured by the hook
            # We pass raw texts and a collate that tokenizes
            ds_test = LitBenchHF(hf_token=hf_token)
            pairs = ds_test.as_text_tuples(split="test")
            # Filter nulls consistent with evaluator
            def _is_bad(s: object) -> bool:
                if s is None:
                    return True
                if not isinstance(s, str):
                    s = str(s)
                t = s.strip()
                return t == "" or t.lower() == "null"
            pairs = [(c, r) for (c, r) in pairs if not _is_bad(c) and not _is_bad(r)]
            texts_eval = []
            for c, r in pairs:
                texts_eval.append(c)
                texts_eval.append(r)
            # Minimal collate using tokenizer
            tok = btrm._tokenizer
            def _collate_eval(batch):
                return tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            # The extractor writes feats_*.pt + manifest.json
            amp_dtype = "bf16" if torch.cuda.is_available() else None
            mounted.extract_features_to_dir(
                texts_eval,
                eval_features_dir,
                batch_size=max(8, encode_batch_size // 2),
                shard_size=20000,
                dtype="float32",
                show_progress=show_progress,
                batch_collate=_collate_eval,
                amp_dtype=amp_dtype,
            )

    # ------------------------------ Evaluate ------------------------------
    logging.info("Evaluating on test split")
    evaluator = PreferenceEvaluator(saerm)
    eval_bs = int(rm_cfg.get("eval_batch_size", encode_batch_size))
    # Optional eval cache: reuse precomputed SAE codes for test set
    eval_cache_path = rm_cfg.get("eval_cache")
    if isinstance(eval_cache_path, str) and os.path.exists(eval_cache_path):
        logging.info("Loading cached eval codes from %s", eval_cache_path)
        try:
            cached = torch.load(eval_cache_path, map_location=device)
            # Expect dict with Z_ch, Z_rj, mu, sig, w optional
            Z_ch = cached.get("Z_ch")
            Z_rj = cached.get("Z_rj")
            if Z_ch is not None and Z_rj is not None:
                # If head/stats present, use them; otherwise, standardize with trained stats
                mu = cached.get("mu") or saerm._mu
                sig = cached.get("sig") or saerm._sig
                w = cached.get("w") or saerm._w
                if mu is not None and sig is not None and w is not None:
                    Z_ch = Z_ch.to(device)
                    Z_rj = Z_rj.to(device)
                    s_ch = ((Z_ch - mu) / sig) @ w
                    s_rj = ((Z_rj - mu) / sig) @ w
                    correct = int((s_ch > s_rj).float().sum().item())
                    total = int(Z_ch.shape[0])
                    acc = float(correct) / max(1, total)
                    class _Tmp:
                        pass
                    tmp = _Tmp()
                    tmp.accuracy = acc
                    tmp.total = total
                    tmp.correct = correct
                    res = tmp
                else:
                    res = evaluator.evaluate(ds, split="test", show_progress=show_progress, batch_size=eval_bs)
            else:
                res = evaluator.evaluate(ds, split="test", show_progress=show_progress, batch_size=eval_bs)
        except Exception as e:
            logging.warning("Failed to use eval cache (%s); falling back to live eval", e)
            res = evaluator.evaluate(ds, split="test", show_progress=show_progress, batch_size=eval_bs)
    else:
        res = evaluator.evaluate(ds, split="test", show_progress=show_progress, batch_size=eval_bs)

    report = {
        "timestamp": int(time.time()),
        "device": str(device),
        "accuracy_test": res.accuracy,
        "total": res.total,
        "correct": res.correct,
        "btrm": {"repo": repo, "max_length": max_length},
        "sae": {"path": sae_dir, "num_neurons": num_neurons, "k_active": k_active},
    }
    with open(os.path.join(outputs_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({"test_accuracy": res.accuracy, "outputs": outputs_dir}, indent=2))


if __name__ == "__main__":
    main()


