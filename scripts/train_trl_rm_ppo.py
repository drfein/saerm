#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from safetensors.torch import load_file as load_safetensors_file
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import cached_file
from trl import PPOConfig, PPOTrainer
import trl.trainer.ppo_trainer as trl_ppo_trainer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from src.nb.downstream.scoring import load_probe_tensor
from src.nb.nullbias.probe import get_score_head, project_to_null_space


class ForwardKwargFilter(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module
        self._accepted_kwargs = set(inspect.signature(module.forward).parameters)

    def forward(self, *args, **kwargs):
        filtered_kwargs = {key: value for key, value in kwargs.items() if key in self._accepted_kwargs}
        if "output_hidden_states" in self._accepted_kwargs:
            filtered_kwargs["output_hidden_states"] = True
        if "return_dict" in self._accepted_kwargs:
            filtered_kwargs["return_dict"] = True
        output = self.module(*args, **filtered_kwargs)
        hidden_states = getattr(output, "hidden_states", None)
        last_hidden_state = getattr(output, "last_hidden_state", None)
        if hidden_states is None and last_hidden_state is not None:
            try:
                output.hidden_states = (last_hidden_state,)
                return output
            except Exception:
                return SimpleNamespace(hidden_states=(last_hidden_state,), last_hidden_state=last_hidden_state)
        return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TRL PPO with a sequence-classification reward model.")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--policy-model", required=True)
    parser.add_argument("--reward-model", required=True)
    parser.add_argument("--probe-file")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-name", default="trl_hh_ppo")
    parser.add_argument("--run-name")
    parser.add_argument("--max-prompt-length", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-mini-batches", type=int, default=1)
    parser.add_argument("--local-rollout-forward-batch-size", type=int, default=1)
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--num-ppo-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--kl-coef", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-grad-norm", type=float, default=0.3)
    parser.add_argument("--stop-token", default="eos")
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--reward-dtype", default="bfloat16")
    parser.add_argument("--policy-dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--reward-load-in-4bit", action="store_true")
    parser.add_argument("--reward-load-in-8bit", action="store_true")
    parser.add_argument("--bnb-4bit-quant-type", default="nf4")
    parser.add_argument("--bnb-4bit-use-double-quant", action="store_true")
    parser.add_argument("--use-peft", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--train-policy-final-layer-only", action="store_true")
    parser.add_argument("--policy-final-layer-patterns", default="lm_head,embed_tokens")
    parser.add_argument("--train-value-final-layer-only", action="store_true")
    parser.add_argument("--value-final-layer-patterns", default="score,classifier,out_proj,head")
    parser.add_argument("--length-penalty-max-len", type=int, default=0)
    parser.add_argument("--length-penalty-ema", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    lowered = str(name).lower()
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp16", "float16", "half"}:
        return torch.float16
    if lowered in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def build_quantization_config(args: argparse.Namespace, torch_dtype: torch.dtype) -> BitsAndBytesConfig | None:
    return build_quantization_config_from_flags(
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        torch_dtype=torch_dtype,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
    )


def build_quantization_config_from_flags(
    *,
    load_in_4bit: bool,
    load_in_8bit: bool,
    torch_dtype: torch.dtype,
    bnb_4bit_quant_type: str,
    bnb_4bit_use_double_quant: bool,
) -> BitsAndBytesConfig | None:
    if load_in_4bit and load_in_8bit:
        raise ValueError("Choose at most one of --load-in-4bit or --load-in-8bit.")
    if load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=torch_dtype,
        )
    if load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def maybe_build_peft_config(args: argparse.Namespace):
    if not args.use_peft:
        return None
    try:
        from peft import LoraConfig, TaskType
    except ImportError as exc:
        raise ImportError("`peft` is required for --use-peft.") from exc
    target_modules = [name.strip() for name in args.lora_target_modules.split(",") if name.strip()]
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
    )


def align_output_head_dtype(model: torch.nn.Module, dtype: torch.dtype) -> None:
    output_head = None
    if hasattr(model, "get_output_embeddings"):
        output_head = model.get_output_embeddings()
    if output_head is not None and hasattr(output_head, "to"):
        output_head.to(dtype=dtype)


def freeze_all_but_patterns(model: torch.nn.Module, patterns_csv: str) -> tuple[int, int, list[str]]:
    normalized = str(patterns_csv).replace(";", ",").replace("|", ",").replace("+", ",")
    patterns = [token.strip().lower() for token in normalized.split(",") if token.strip()]
    if not patterns:
        raise ValueError("Expected at least one final-layer pattern.")
    trainable = 0
    total = 0
    matched_names: list[str] = []
    for name, param in model.named_parameters():
        keep = any(pattern in name.lower() for pattern in patterns)
        param.requires_grad = keep
        count = int(param.numel())
        total += count
        if keep:
            trainable += count
            matched_names.append(name)
    return trainable, total, matched_names


def render_prompt(tokenizer: Any, row: dict[str, Any], max_prompt_length: int) -> dict[str, list[int]]:
    messages = row.get("prompt")
    if isinstance(messages, list) and messages:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            truncation=True,
            max_length=max_prompt_length,
        )
        attention_mask = [1] * len(rendered)
        return {"input_ids": rendered, "attention_mask": attention_mask}

    question = row.get("question")
    if not isinstance(question, str) or not question.strip():
        question = str(row.get("prompt", "")).strip()
    if not question:
        raise ValueError("Record is missing a usable prompt/question")

    if tokenizer.chat_template is not None:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=True,
            add_generation_prompt=True,
            truncation=True,
            max_length=max_prompt_length,
        )
        attention_mask = [1] * len(rendered)
        return {"input_ids": rendered, "attention_mask": attention_mask}

    rendered = tokenizer(
        question,
        truncation=True,
        max_length=max_prompt_length,
        add_special_tokens=True,
    )
    return {"input_ids": rendered["input_ids"], "attention_mask": rendered["attention_mask"]}


def load_prompt_dataset(path: str, tokenizer: Any, max_prompt_length: int) -> Dataset:
    dataset = load_dataset("json", data_files=path, split="train")
    column_names = list(dataset.column_names)

    def tokenize_row(row: dict[str, Any]) -> dict[str, list[int]]:
        rendered = render_prompt(tokenizer, row, max_prompt_length=max_prompt_length)
        if not rendered["input_ids"]:
            raise ValueError("Prompt tokenization produced an empty prompt")
        return rendered

    dataset = dataset.map(tokenize_row, remove_columns=column_names)
    return dataset


class NullingSequenceClassificationReward(torch.nn.Module):
    def __init__(
        self,
        model_path: str,
        *,
        probe_path: str | None,
        alpha: float,
        torch_dtype: torch.dtype,
        attn_implementation: str | None,
        trust_remote_code: bool,
        quantization_config: BitsAndBytesConfig | None,
    ) -> None:
        super().__init__()
        self.model = self._load_reward_model(
            model_path=model_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            quantization_config=quantization_config,
        )
        original_base_model_prefix = self.model.base_model_prefix
        self.backbone = ForwardKwargFilter(getattr(self.model, original_base_model_prefix))
        self.base_model_prefix = "backbone"
        if self.model.config.pad_token_id is None and self.model.config.eos_token_id is not None:
            self.model.config.pad_token_id = self.model.config.eos_token_id
        self.model_path = model_path
        self.score_module = get_score_head(self.model)

        probe = load_probe_tensor(probe_path)
        if probe is None:
            self.register_buffer("probe", torch.empty(0), persistent=False)
            self.has_probe = False
        else:
            self.register_buffer("probe", probe.float(), persistent=False)
            self.has_probe = True
        score_weight, score_bias = self._load_score_head_from_checkpoint()
        self.register_buffer("checkpoint_score_weight", score_weight, persistent=False)
        self.register_buffer("checkpoint_score_bias", score_bias, persistent=False)
        self.alpha = float(alpha)

    @staticmethod
    def _load_reward_model(
        *,
        model_path: str,
        trust_remote_code: bool,
        torch_dtype: torch.dtype,
        attn_implementation: str | None,
        quantization_config: BitsAndBytesConfig | None,
    ) -> AutoModelForSequenceClassification:
        common_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "torch_dtype": torch_dtype,
            "num_labels": 1,
            "ignore_mismatched_sizes": True,
        }
        if quantization_config is not None:
            common_kwargs["quantization_config"] = quantization_config
        if attn_implementation:
            common_kwargs["attn_implementation"] = attn_implementation

        adapter_config = Path(model_path) / "adapter_config.json"
        if adapter_config.exists():
            try:
                from peft import AutoPeftModelForSequenceClassification
            except ImportError as exc:
                raise ImportError(
                    "Reward model path contains a PEFT adapter; install `peft` to load it."
                ) from exc

            try:
                peft_model = AutoPeftModelForSequenceClassification.from_pretrained(model_path, **common_kwargs)
            except (ValueError, RuntimeError) as exc:
                message = str(exc)
                if (
                    attn_implementation == "sdpa"
                    and "does not support an attention implementation through torch.nn.functional.scaled_dot_product_attention" in message
                ):
                    fallback_kwargs = dict(common_kwargs)
                    fallback_kwargs["attn_implementation"] = "eager"
                    print(f"[reward-model] retrying {model_path} with attn_implementation=eager")
                    peft_model = AutoPeftModelForSequenceClassification.from_pretrained(model_path, **fallback_kwargs)
                else:
                    raise
            if hasattr(peft_model, "merge_and_unload"):
                return peft_model.merge_and_unload()
            return peft_model

        model_kwargs = dict(common_kwargs)
        try:
            return AutoModelForSequenceClassification.from_pretrained(model_path, **model_kwargs)
        except (ValueError, RuntimeError) as exc:
            message = str(exc)
            if (
                attn_implementation == "sdpa"
                and "does not support an attention implementation through torch.nn.functional.scaled_dot_product_attention" in message
            ):
                fallback_kwargs = dict(model_kwargs)
                fallback_kwargs["attn_implementation"] = "eager"
                print(f"[reward-model] retrying {model_path} with attn_implementation=eager")
                return AutoModelForSequenceClassification.from_pretrained(model_path, **fallback_kwargs)
            raise

    def _load_score_head_from_checkpoint(self) -> tuple[torch.Tensor, torch.Tensor]:
        head_names = ("score", "classifier", "out_proj", "head")
        index_file = cached_file(
            self.model_path,
            "model.safetensors.index.json",
            local_files_only=True,
            _raise_exceptions_for_missing_entries=False,
            _raise_exceptions_for_gated_repo=False,
            _raise_exceptions_for_connection_errors=False,
        )
        if index_file is not None:
            with Path(index_file).open("r", encoding="utf-8") as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
            for stem in head_names:
                weight_key = f"{stem}.weight"
                shard_name = weight_map.get(weight_key)
                if shard_name is None:
                    continue
                shard_path = cached_file(
                    self.model_path,
                    shard_name,
                    local_files_only=True,
                    _raise_exceptions_for_missing_entries=False,
                    _raise_exceptions_for_gated_repo=False,
                    _raise_exceptions_for_connection_errors=False,
                )
                if shard_path is None:
                    continue
                shard_tensors = load_safetensors_file(str(shard_path))
                bias = shard_tensors.get(f"{stem}.bias")
                return (
                    shard_tensors[weight_key].detach().clone(),
                    torch.empty(0) if bias is None else bias.detach().clone(),
                )

        single_file = cached_file(
            self.model_path,
            "model.safetensors",
            local_files_only=True,
            _raise_exceptions_for_missing_entries=False,
            _raise_exceptions_for_gated_repo=False,
            _raise_exceptions_for_connection_errors=False,
        )
        if single_file is not None:
            tensors = load_safetensors_file(str(single_file))
            for stem in head_names:
                weight = tensors.get(f"{stem}.weight")
                if weight is None:
                    continue
                bias = tensors.get(f"{stem}.bias")
                return (
                    weight.detach().clone(),
                    torch.empty(0) if bias is None else bias.detach().clone(),
                )

        return torch.empty(0), torch.empty(0)

    def _score_via_named_parameters(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        named_parameters = dict(self.model.named_parameters())
        for stem in ("score", "classifier", "out_proj", "head"):
            weight = named_parameters.get(f"{stem}.weight")
            if weight is None:
                continue
            bias = named_parameters.get(f"{stem}.bias")
            return F.linear(hidden_states, weight, bias)
        return None

    def score(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.has_probe:
            hidden_states = project_to_null_space(hidden_states, self.probe, alpha=self.alpha)
        if self.score_module is not None:
            return self.score_module(hidden_states)
        score_via_params = self._score_via_named_parameters(hidden_states)
        if score_via_params is not None:
            return score_via_params
        if self.checkpoint_score_weight.numel() > 0:
            bias = None if self.checkpoint_score_bias.numel() == 0 else self.checkpoint_score_bias
            return F.linear(hidden_states, self.checkpoint_score_weight, bias)
        return get_score_head(self.model)(hidden_states)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def build_ppo_config(args: argparse.Namespace) -> PPOConfig:
    signature = inspect.signature(PPOConfig.__init__)
    policy_dtype_name = str(args.policy_dtype).lower()
    use_bf16 = policy_dtype_name in {"bf16", "bfloat16"}
    use_fp16 = policy_dtype_name in {"fp16", "float16", "half"}
    config_kwargs: dict[str, Any] = {
        "output_dir": str(Path(args.output_dir).resolve()),
        "run_name": args.run_name or args.experiment_name,
        "exp_name": args.experiment_name,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_mini_batches": args.num_mini_batches,
        "total_episodes": args.train_steps
        * args.per_device_train_batch_size
        * args.gradient_accumulation_steps,
        "num_ppo_epochs": args.num_ppo_epochs,
        "local_rollout_forward_batch_size": args.local_rollout_forward_batch_size,
        "response_length": args.max_new_tokens,
        "temperature": args.temperature,
        "logging_steps": args.logging_steps,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "bf16": use_bf16,
        "gradient_checkpointing": not args.no_gradient_checkpointing,
        "report_to": [] if args.report_to == "none" else [args.report_to],
        "remove_unused_columns": False,
        "num_sample_generations": 0,
        "kl_coef": args.kl_coef,
        "seed": args.seed,
        "disable_tqdm": False,
    }
    if "fp16" in signature.parameters:
        config_kwargs["fp16"] = use_fp16
    if "max_grad_norm" in signature.parameters:
        config_kwargs["max_grad_norm"] = args.max_grad_norm
    if "stop_token" in signature.parameters:
        config_kwargs["stop_token"] = args.stop_token
    if "eval_strategy" in signature.parameters:
        config_kwargs["eval_strategy"] = "steps"
    elif "evaluation_strategy" in signature.parameters:
        config_kwargs["evaluation_strategy"] = "steps"
    if "save_strategy" in signature.parameters:
        config_kwargs["save_strategy"] = "steps"
    if "logging_strategy" in signature.parameters:
        config_kwargs["logging_strategy"] = "steps"
    return PPOConfig(**config_kwargs)


def patch_trl_generation_config(args: argparse.Namespace) -> None:
    base_generation_config = trl_ppo_trainer.GenerationConfig

    class SafeGenerationConfig(base_generation_config):
        def __init__(self, *gen_args, **gen_kwargs):
            gen_kwargs["temperature"] = max(float(gen_kwargs.get("temperature", args.temperature)), 1e-5)
            gen_kwargs["top_p"] = float(args.top_p)
            gen_kwargs["remove_invalid_values"] = True
            gen_kwargs["renormalize_logits"] = True
            super().__init__(*gen_args, **gen_kwargs)

    trl_ppo_trainer.GenerationConfig = SafeGenerationConfig


def patch_trl_reward_with_length_penalty(args: argparse.Namespace) -> None:
    if args.length_penalty_max_len <= 0:
        return

    original_get_reward = trl_ppo_trainer.get_reward
    state = {"sigma_ema": None}
    max_len = float(args.length_penalty_max_len)
    ema = float(args.length_penalty_ema)

    def get_reward_with_length_penalty(model, query_responses, pad_token_id, context_length):
        reward_logits, final_rewards, sequence_lengths = original_get_reward(
            model,
            query_responses,
            pad_token_id,
            context_length,
        )
        batch_std = final_rewards.std(unbiased=False)
        if state["sigma_ema"] is None:
            state["sigma_ema"] = batch_std.detach()
        else:
            state["sigma_ema"] = (ema * state["sigma_ema"]) + ((1.0 - ema) * batch_std.detach())

        response_lengths = (sequence_lengths - context_length + 1).clamp_min(0).to(final_rewards.dtype)
        penalty = (1.0 - (response_lengths / max_len)) * state["sigma_ema"].to(final_rewards.dtype)
        adjusted = final_rewards + penalty
        return reward_logits, adjusted, sequence_lengths

    trl_ppo_trainer.get_reward = get_reward_with_length_penalty


def patch_trl_rollout_length_logging() -> None:
    original_get_reward = trl_ppo_trainer.get_reward
    state = {"last_mean_response_length": None}

    def get_reward_with_logging(model, query_responses, pad_token_id, context_length):
        reward_logits, final_rewards, sequence_lengths = original_get_reward(
            model,
            query_responses,
            pad_token_id,
            context_length,
        )
        response_lengths = (sequence_lengths - context_length + 1).clamp_min(0)
        state["last_mean_response_length"] = float(response_lengths.float().mean().item())
        return reward_logits, final_rewards, sequence_lengths

    trl_ppo_trainer.get_reward = get_reward_with_logging

    original_log = PPOTrainer.log

    def log_with_rollout_length(self, logs, *args, **kwargs):
        if state["last_mean_response_length"] is not None:
            logs = dict(logs)
            logs.setdefault("rollout/mean_response_length", state["last_mean_response_length"])
        return original_log(self, logs, *args, **kwargs)

    PPOTrainer.log = log_with_rollout_length


def patch_trl_policy_value_wrapper_checkpointing() -> None:
    wrapper_cls = getattr(trl_ppo_trainer, "PolicyAndValueWrapper", None)
    if wrapper_cls is None:
        return

    def _call_checkpointing_method(instance: Any, method_name: str) -> None:
        for attr in ("policy", "policy_model", "model", "pretrained_model", "value_model", "critic"):
            child = getattr(instance, attr, None)
            if child is not None and hasattr(child, method_name):
                try:
                    getattr(child, method_name)()
                except AttributeError as exc:
                    # Some HF model variants can miss the internal hook when TRL toggles
                    # checkpointing around generation. This is safe to ignore.
                    if "_require_grads_hook" in str(exc) and method_name == "gradient_checkpointing_disable":
                        continue
                    raise

    if not hasattr(wrapper_cls, "gradient_checkpointing_disable"):
        def gradient_checkpointing_disable(self) -> None:
            _call_checkpointing_method(self, "gradient_checkpointing_disable")

        wrapper_cls.gradient_checkpointing_disable = gradient_checkpointing_disable

    if not hasattr(wrapper_cls, "gradient_checkpointing_enable"):
        def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict[str, Any] | None = None) -> None:
            del gradient_checkpointing_kwargs
            _call_checkpointing_method(self, "gradient_checkpointing_enable")

        wrapper_cls.gradient_checkpointing_enable = gradient_checkpointing_enable


def main() -> None:
    args = parse_args()
    patch_trl_generation_config(args)
    patch_trl_reward_with_length_penalty(args)
    patch_trl_rollout_length_logging()
    patch_trl_policy_value_wrapper_checkpointing()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    policy_dtype = dtype_from_name(args.policy_dtype)
    reward_dtype = dtype_from_name(args.reward_dtype)
    policy_quantization_config = build_quantization_config(args, policy_dtype)
    reward_quantization_config = build_quantization_config_from_flags(
        load_in_4bit=args.reward_load_in_4bit or args.load_in_4bit,
        load_in_8bit=args.reward_load_in_8bit or args.load_in_8bit,
        torch_dtype=reward_dtype,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
    )
    peft_config = maybe_build_peft_config(args)
    if peft_config is not None and (args.train_policy_final_layer_only or args.train_value_final_layer_only):
        raise ValueError("Final-layer-only training is incompatible with --use-peft in this script.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.policy_model,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    train_dataset = load_prompt_dataset(args.train_file, tokenizer, max_prompt_length=args.max_prompt_length)
    eval_dataset = load_prompt_dataset(args.eval_file, tokenizer, max_prompt_length=args.max_prompt_length)

    common_model_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": policy_dtype,
    }
    if policy_quantization_config is not None:
        common_model_kwargs["quantization_config"] = policy_quantization_config
    if args.attn_implementation:
        common_model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(args.policy_model, **common_model_kwargs)
    align_output_head_dtype(model, policy_dtype)
    if peft_config is not None and policy_quantization_config is not None:
        try:
            from peft import prepare_model_for_kbit_training
        except ImportError as exc:
            raise ImportError("`peft` is required for quantized PEFT PPO.") from exc
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=not args.no_gradient_checkpointing,
        )
        align_output_head_dtype(model, policy_dtype)
    ref_model = None if peft_config is not None else AutoModelForCausalLM.from_pretrained(args.policy_model, **common_model_kwargs)
    if ref_model is not None:
        align_output_head_dtype(ref_model, policy_dtype)
    value_model = AutoModelForSequenceClassification.from_pretrained(
        args.policy_model,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=policy_dtype,
        quantization_config=policy_quantization_config,
        attn_implementation=args.attn_implementation,
        num_labels=1,
        ignore_mismatched_sizes=True,
    )
    reward_model = NullingSequenceClassificationReward(
        args.reward_model,
        probe_path=args.probe_file,
        alpha=args.alpha,
        torch_dtype=reward_dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
        quantization_config=reward_quantization_config,
    )

    if args.train_policy_final_layer_only:
        policy_trainable, policy_total, policy_names = freeze_all_but_patterns(model, args.policy_final_layer_patterns)
        if policy_trainable <= 0:
            raise ValueError(
                f"No policy parameters matched --policy-final-layer-patterns={args.policy_final_layer_patterns!r}."
            )
        print(
            f"Policy final-layer-only mode: trainable={policy_trainable} / total={policy_total} "
            f"({100.0 * policy_trainable / max(policy_total, 1):.4f}%)."
        )
        print(f"Policy matched parameters ({len(policy_names)}): {policy_names}")

    if args.train_value_final_layer_only:
        value_trainable, value_total, value_names = freeze_all_but_patterns(value_model, args.value_final_layer_patterns)
        if value_trainable <= 0:
            raise ValueError(
                f"No value parameters matched --value-final-layer-patterns={args.value_final_layer_patterns!r}."
            )
        print(
            f"Value final-layer-only mode: trainable={value_trainable} / total={value_total} "
            f"({100.0 * value_trainable / max(value_total, 1):.4f}%)."
        )
        print(f"Value matched parameters ({len(value_names)}): {value_names}")

    configured_models = [model, value_model, reward_model.model]
    if ref_model is not None:
        configured_models.append(ref_model)
    for cfg_model in configured_models:
        if getattr(cfg_model.config, "pad_token_id", None) is None:
            cfg_model.config.pad_token_id = tokenizer.pad_token_id
        cfg_model.config.use_cache = False

    config = build_ppo_config(args)
    trainer = PPOTrainer(
        args=config,
        processing_class=tokenizer,
        model=model,
        ref_model=ref_model,
        reward_model=reward_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        value_model=value_model,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir / "final"))

    summary = {
        "train_rows": len(train_dataset),
        "eval_rows": len(eval_dataset),
        "policy_model": args.policy_model,
        "reward_model": args.reward_model,
        "probe_file": args.probe_file,
        "output_dir": str(output_dir.resolve()),
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
