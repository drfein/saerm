#!/usr/bin/env python3
import argparse
import json
import os
import time

import torch

from saerm.rm.btrm import BTRM
from saerm.rm.saerm import SAERM
from saerm.process_data.litbench_hf import LitBenchHF
from saerm.eval.preference import PreferenceEvaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved SAERM head on LitBench test set (YAML-configured).")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    try:
        import yaml  # type: ignore
    except Exception as e:
        raise SystemExit("PyYAML is required. Install with: pip install pyyaml") from e

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    btrm_cfg = cfg.get("btrm", {})
    eval_cfg = cfg.get("eval", {})
    saerm_cfg = cfg.get("saerm", {})

    # .env support for HF_TOKEN
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except Exception:
        pass

    saerm_dir = saerm_cfg["dir"]
    repo = btrm_cfg["repo"]
    hf_token = os.environ.get("HF_TOKEN", "") or btrm_cfg.get("hf_token") or ""
    device = btrm_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    max_length = int(btrm_cfg["max_length"])
    encode_batch_size = int(eval_cfg.get("encode_batch_size", 32))
    show_progress = bool(eval_cfg.get("show_progress", True))

    btrm = BTRM.load(
        repo,
        {
            "hf_token": hf_token,
            "max_length": max_length,
            "device": device,
            "trust_remote_code": False,
        },
    )
    device_t = btrm._model.device  # type: ignore[attr-defined]

    tok = btrm._tokenizer

    def encode_fn(texts):
        return tok(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device_t)

    saerm = SAERM.load(saerm_dir, {"device": str(device_t), "encode_fn": encode_fn})

    ds = LitBenchHF(hf_token=hf_token)
    evaluator = PreferenceEvaluator(saerm)
    res = evaluator.evaluate(ds, split="test", show_progress=show_progress)

    report = {
        "timestamp": int(time.time()),
        "device": str(device_t),
        "accuracy_test": res.accuracy,
        "total": res.total,
        "correct": res.correct,
        "saerm_dir": saerm_dir,
        "btrm": {"repo": repo, "max_length": max_length},
    }
    out_path = os.path.join(saerm_dir, "eval_test_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({"test_accuracy": res.accuracy, "report": out_path}, indent=2))


if __name__ == "__main__":
    main()


