#!/usr/bin/env python3
import argparse
import json
import os
import time
import logging

import torch

from saerm.rm.btrm import BTRM
from saerm.sae.mounted import MountedSAE
from saerm.rm.xgbrm import XGBSAERM
from saerm.process_data.litbench_hf import LitBenchHF
from saerm.eval.preference import PreferenceEvaluator


def _repo_subdir(repo: str) -> str:
    parts = [p for p in str(repo).split("/") if p]
    return os.path.join(*parts) if parts else ""

def _select_last_token(hidden_states: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    idx = attention_mask.sum(dim=1) - 1
    idx = torch.clamp(idx, min=0)
    b = hidden_states.size(0)
    return hidden_states[torch.arange(b, device=hidden_states.device), idx, :]


def _activation_transform_from_output(output):
    hidden_states = getattr(output, "hidden_states", None)
    if hidden_states is None:
        raise RuntimeError("Model output does not contain hidden_states. Ensure config.output_hidden_states=True.")
    model = output.__dict__.get("_saerm_attached_model", None)
    if model is None or not hasattr(model, "_saerm_last_batch"):
        last_h = hidden_states[-1]
        return last_h.mean(dim=1)
    batch = model._saerm_last_batch
    last_h = hidden_states[-1]
    return _select_last_token(last_h, batch["input_ids"], batch["attention_mask"]).float()


def main():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Train an XGB ranker head on SAE codes and report test accuracy (YAML-configured).")
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

    os.makedirs(outputs_dir, exist_ok=True)

    # Resolve per-model features directory if provided
    sae_features_dir = None
    if isinstance(sae_features_root, str) and sae_features_root.strip() != "":
        sae_features_dir = os.path.join(sae_features_root, _repo_subdir(repo))

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

    # Attach batch capture wrapper (same as train_sae)
    orig_forward = btrm._model.forward  # type: ignore[attr-defined]
    def forward_with_batch_capture(*fw_args, **fw_kwargs):  # type: ignore[no-redef]
        out = orig_forward(*fw_args, **fw_kwargs)
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

    # Load SAE weights
    logging.info("Loading SAE from %s", sae_dir)
    sae_ckpt_path = os.path.join(sae_dir, "sae.pt")
    ckpt = torch.load(sae_ckpt_path, map_location=btrm._model.device)  # type: ignore[attr-defined]
    num_neurons = int(ckpt.get("num_neurons"))
    k_active = int(ckpt.get("k_active"))

    mounted = MountedSAE(
        base_model=btrm._model,
        layer=btrm._model,
        num_neurons=num_neurons,
        k_active=k_active,
        device=str(btrm._model.device),  # type: ignore[attr-defined]
        flatten=False,
        activation_transform=_activation_transform_from_output,
    )
    # Initialize SAE dims
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

    # ------------------------ Build encode function ------------------------
    tok = btrm._tokenizer
    device = btrm._model.device  # type: ignore[attr-defined]
    def encode_fn(texts):
        return tok(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)

    # XGBSAERM head
    xgb_params = rm_cfg.get("xgb_params") or None
    model = XGBSAERM(sae=mounted, encode_fn=encode_fn, device=str(device), encode_batch_size=encode_batch_size, xgb_params=xgb_params)

    # ------------------------------ Data & Train ------------------------------
    logging.info("Loading preference dataset (LitBenchHF)")
    ds = LitBenchHF(hf_token=hf_token)

    if sae_features_dir and os.path.exists(os.path.join(sae_features_dir, "manifest.json")):
        logging.info("Found precomputed features in %s; training XGB head from features", sae_features_dir)
        with open(os.path.join(sae_features_dir, "manifest.json"), "r") as f:
            manifest = json.load(f)
        shards = manifest.get("shards", [])
        Z_ch_list = []
        Z_rj_list = []
        sae_inst = mounted.sae
        if sae_inst is None:
            raise RuntimeError("SAE instance missing after load")
        sae_device = sae_inst.device  # type: ignore[attr-defined]
        step = max(1, encode_batch_size)
        row_cursor = 0
        for sh in shards:
            path = os.path.join(sae_features_dir, sh["file"])  # type: ignore[index]
            feats_cpu = torch.load(path, map_location="cpu").float()
            for i in range(0, feats_cpu.shape[0], step):
                chunk = feats_cpu[i : i + step].to(sae_device)
                with torch.no_grad():
                    _xhat, info = sae_inst(chunk)
                    codes = info["codes"].detach().to("cpu")
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
            raise RuntimeError("No codes extracted from features; cannot train XGB head")
        Z_ch = torch.cat(Z_ch_list, dim=0).to(device)
        Z_rj = torch.cat(Z_rj_list, dim=0).to(device)
        n_pairs = min(Z_ch.shape[0], Z_rj.shape[0])
        if Z_ch.shape[0] != Z_rj.shape[0]:
            logging.warning("Mismatched chosen/rejected counts: %d vs %d; truncating to %d", Z_ch.shape[0], Z_rj.shape[0], n_pairs)
            Z_ch = Z_ch[:n_pairs]
            Z_rj = Z_rj[:n_pairs]
        model.train_from_codes(Z_ch, Z_rj)
        logging.info("XGB head trained from features: pairs=%d, dim=%d", int(n_pairs), int(Z_ch.shape[1]))
    else:
        model.train(ds)

    # Save head and stats
    logging.info("Saving XGB head and stats to %s", outputs_dir)
    model.save(outputs_dir)

    # ------------------------------ Evaluate ------------------------------
    logging.info("Evaluating on test split")
    evaluator = PreferenceEvaluator(model)
    eval_bs = int(rm_cfg.get("eval_batch_size", encode_batch_size))
    res = evaluator.evaluate(ds, split="test", show_progress=show_progress, batch_size=eval_bs)

    report = {
        "timestamp": int(time.time()),
        "device": str(device),
        "accuracy_test": res.accuracy,
        "total": res.total,
        "correct": res.correct,
        "btrm": {"repo": repo, "max_length": max_length},
        "sae": {"path": sae_dir, "num_neurons": num_neurons, "k_active": k_active},
        "xgb_params": xgb_params or {},
    }
    with open(os.path.join(outputs_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({"test_accuracy": res.accuracy, "outputs": outputs_dir}, indent=2))


if __name__ == "__main__":
    main()



