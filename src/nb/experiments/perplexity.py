
"""
RMs
Skywork-Reward-V2-Llama-3.1-8B --base--> meta-llama/Llama-3.1-8B-Instruct https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-8B
Skywork-Reward-V2-Qwen3-8B --base--> Qwen/Qwen3-8B
Skywork-Reward-V2-Qwen3-0.6 --base--> Qwen/Qwen3-0.6B
Allen-Llama-3.1-8B-Instruct-RM-RB2 --base--> meta-llama/Llama-3.1-8B-Instruct https://huggingface.co/allenai/Llama-3.1-8B-Instruct-RM-RB2

Models for perplexity (need good range)
google/gemma-3-12b-it               # Different family
google/gemma-2-9b-it?
meta-llama/Llama-2-7b-chat-hf       # Different generation
meta-llama/Llama-3.1-8B-Instruct    # Basis for two tested RMs
Qwen/Qwen3-8B                       # Basis for one tested RM each
Qwen/Qwen3-0.6B
Qwen/Qwen2.5-7B-Instruct            # Different generation

    Don't test actual base models of base of RM? Maybe add them if panel subtraction not representative enough? Not sure.
    google/gemma-3-12b-pt
    meta-llama/Llama-2-7b-hf
    meta-llama/Llama-3.1-8B
    Qwen/Qwen3-8B-Base
    Qwen/Qwen3-0.6B-Base

Q1: Use instruction-tuned (it) or pre-trained (pt) models? Probably pt or both, right?
    Apparently, RMs were all traiend on instruct models anyway... Probabyl no meaning in testing all-the-way-down base models

Q2: Double check: Evidence for using same model actually improves RLHF, which is what we use to motivate our experiment
    [RewardBench2] reports that for PPO-style RLHF, whether the reward model and policy come from the same model lineage is an important factor; using an RM from a different base family can make downstream RLHF performance “degrade significantly.”
    They also report strong drops when there's misalignment between the policy and the RM's base model (and/or when RM training prompts are out of distribution relative to RL prompts).
    This is a pretty direct confirmation of the phenomenon, but better framed as “same lineage/starting point” rather than “same exact model.”

Q3: Is looking for a "privileged reward" a reasonable idea to begin with?
    Better framed as in-distribution advantage + inherited inductive biases (privelege could sound intentional?) "Does the RM's score depend on similarity-to-base-model after controlling for actual human preference/quality?"
    We also observe a lot of interesting quirks for models based on which base model was used in our experiments, e.g., which bsae model is better in math seems to be inherited by RM model. [CHECK]
    In short, yes.
    Reason against: Practially speaking, if you are training a product-ready language model, you probably already have a base model and a preference dataset and will just use that anyways. Main reason thus remains for general open source community and using RMs for other purposes (guide RL red-teaming)

Q4: What is a good metric to use to measure similarity? Perplexity, KL, embedding cosine?
    Perplexity can work well, BUT, it also measures "genericness" of sequences and can differ between models/tokenizers and it also does not exclusively measure how "model-like" a sequence is
    first, normalize: for a promt x and compeltion y

    s_model(x, y) = 1/(|y|) * sum_t log P_model(y_t | x, y_<t)

    second, compute model-specific part of perplexity via subtracting average over models M (excluding base model) "panel-relative perplexity":

    s_delta(x, y) = s_base - 1/|M| * sum_m s_model_m(x, y) for m in M\{base}

    --> calculate r(x,y) and s_delta(x,y) for all (x,y) in dataset and compute correlatoins (or just scatter plot for start)

    Because I want to compare across different familites and tokenizers, I should probably norm s_model(x,y) by character number of y or bytes in y rather than number of tokens |y|. Or does that even matter? Maybe collect all during data generation.

Q5: Assuming we find a correlation, what would this imply? Is it the "in-distribution advantage + inherited inductive biases" effect we want or could it also imply something else? Would it be sufficient to justify our hypothesis?
    --> This analysis will only work if models are generally capable. If they are not, then the perplexity and reward differences may just reflect general capability differences rather than similarity to base model.
    --> Correlation could be produced if one of the models (base) was used to generate synthetic data for RM, so it learns that style would be on-distirbutions. I think we can still say this is bad.
    --> Need to look out for confounding factors like length/verbosity, refusal style/format, general formatting, difficulty of topic, tokenizer-specifics (negligble here?), biased panel.

    Claims that could be made:
    - RLHF will partially optimzie toward base-family style/anifold features or produce less helpful rewards to guide RLHF, potentially explaining observation
    - RM performance evals could be contaminated and results depend on which model you use to construct synthetic dataset

Q6: Is model selection sufficient?
    Consider adding base models to remove "instruction" part of perplexity?

Q7: What dataset to use here? What model generations can or must be included in the dataset?
    Just use a concatenated version of what we had so far? --> Probably not good, as parts synthetically generated by specific models. Could bias results.
    This is hard. Technically, using the fewer promtps with more completions should make analysis better.
    Could use allenai/tulu-3-wildchat-reused-on-policy-8b, which includes a lot of different model completions https://huggingface.co/datasets/allenai/tulu-3-wildchat-reused-on-policy-8b/viewer/default/train?p=166
        or allenai/WildChat-4.8M https://huggingface.co/datasets/allenai/WildChat-4.8M
    Think more about this. Having out-of-distribution generations should be good, as it should give more signal/variance, but also need on-distribution generations.

    tulu-3-wildhcat-reused-on-policy-8b has generations from: gemma-2-9b, Qwen2.5-7B, llama-3.1-8b, but none from the other base models. For now, I think this is fine.


Todo:
- Implement perplexity calculation for different model families/tokenizers
    - Probably make a base class that handles different families
- Load dataset and calculate perplexities + norms (char, bytes, tokens)
    - Store prompt_id, completion_id, generator_model, prompt_text, completion_text, plus metadata (if all fo them are available)
    - Store all in a dataset
    - Run separately for each model
    - Fix random seed here when sampling data
- Load dataset, use same UID of prompt completions, and feed through RMs
- Write analysis code and create plots
    - Add correlation calculations
    - Potentially try different perpelxity normalizations for robustness
    - Do I need to normalize RM scores? I don't think so, as I am only picking "largest correlation" for each RM independently, right?
    - Potentially add color to scatter dots depending on response token/byte length?
- Profit, hopefully
"""

from __future__ import annotations

import pandas as pd
from datasets import load_dataset
from dataclasses import dataclass
from typing import Dict, Optional, Union, List, Any
import math
import torch
from tqdm import tqdm
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
)

# load allenai/tulu-3-wildchat-reused-on-policy-8b dataset
def load_dataset_local(dataset_name: str):
    """"""
    dataset = load_dataset(dataset_name, split="train")
    return dataset

@dataclass
class ModelHandle:
    model_id: str
    model: torch.nn.Module
    tok: Any  # AutoTokenizer OR AutoProcessor (for Gemma 3)
    is_processor: bool  # True => use tok as processor, not tokenizer


# Implement perplexity calculation for different model families/tokenizers
class PerplexityEvaluator:
    """"""
    def __init__(self, model_name: str):
        self.model_name = model_name
        # Load model and tokenizer here based on model_name
        self.model_handle = self._load_model(
            model_name,
            device_map="auto",
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    def _maybe_set_pad_token(self, tokenizer: AutoTokenizer) -> None:
        # Many decoder-only models don't define a pad token; set it for batching.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    def _load_model(
            self,
            model_id: str,
            *,
            device_map: str = "auto",
            dtype: torch.dtype = torch.bfloat16,
            trust_remote_code: bool = False,
        ):

        handle = None
        # Need this?
        quant_cfg = None

        # Gemma 3 is multimodal; HF usage shows AutoProcessor + Gemma3ForConditionalGeneration :contentReference[oaicite:4]{index=4}.
        # We'll still score text-only by passing just text tokens.
        if model_id.startswith("google/gemma-3"):
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=device_map,
                dtype=dtype,
                quantization_config=quant_cfg,
                trust_remote_code=trust_remote_code,
            ).eval()
            handle = ModelHandle(model_id, model, processor, is_processor=True)
        else:
            tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=trust_remote_code)
            self._maybe_set_pad_token(tok)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=device_map,
                dtype=dtype,
                quantization_config=quant_cfg,
                trust_remote_code=trust_remote_code,
            ).eval()
            handle = ModelHandle(model_id, model, tok, is_processor=False)

        return handle

##
    @staticmethod
    def split_messages_for_scoring(messages):
        """
        messages: List[{"role": ..., "content": ...}]
        Returns: (prompt_messages, full_messages, completion_text)
        Scores ONLY the last assistant message content.
        """
        if not isinstance(messages, list):
            raise TypeError(f"Expected list of messages, got {type(messages)}")

        # Find last assistant
        last_asst_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                last_asst_idx = i
                break
        if last_asst_idx is None:
            raise ValueError("No assistant message found to score.")

        prompt_messages = messages[:last_asst_idx]
        full_messages = messages[: last_asst_idx + 1]
        completion_text = messages[last_asst_idx].get("content", "")

        return prompt_messages, full_messages, completion_text

    def _encode_text(self, handle: ModelHandle, text: str):
        """
        Returns (input_ids, attention_mask) on CPU.
        """
        if handle.is_processor:
            enc = handle.tok(text=text, return_tensors="pt")
        else:
            enc = handle.tok(text, return_tensors="pt")

        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))
        return input_ids, attention_mask

    def _chat_to_text(self, handle: ModelHandle, messages, add_generation_prompt: bool):
        """
        Returns a chat-formatted string using the appropriate tokenizer/processor.
        """
        return handle.tok.apply_chat_template(
            messages,
            tokenize=False,                 # IMPORTANT: always string
            add_generation_prompt=add_generation_prompt,
        )

    def _encode_prompt_and_full(self, handle: ModelHandle, prompt, completion, use_chat_template: bool = True):
        """
        Returns: input_ids, attention_mask, prompt_len, completion_text
        """
        # Case A: dataset row format (chosen/rejected are message lists)
        if isinstance(completion, list):
            prompt_msgs, full_msgs, completion_text = self.split_messages_for_scoring(completion)

            if use_chat_template and hasattr(handle.tok, "apply_chat_template"):
                prompt_text = self._chat_to_text(handle, prompt_msgs, add_generation_prompt=True)
                full_text   = self._chat_to_text(handle, full_msgs,  add_generation_prompt=False)
            else:
                # fallback: just concatenate contents (not ideal, but avoids crashing)
                prompt_text = "".join(m.get("content", "") for m in prompt_msgs)
                full_text   = "".join(m.get("content", "") for m in full_msgs)

            prompt_ids, _ = self._encode_text(handle, prompt_text)
            full_ids, attn = self._encode_text(handle, full_text)
            prompt_len = prompt_ids.shape[1]
            return full_ids, attn, prompt_len, completion_text

        # Case B: original string prompt/completion
        if not (isinstance(prompt, str) and isinstance(completion, str)):
            raise TypeError(f"Expected (str, str) or (any, list[dict]); got ({type(prompt)}, {type(completion)})")

        completion_text = completion

        if use_chat_template and hasattr(handle.tok, "apply_chat_template") and (not handle.is_processor) and getattr(handle.tok, "chat_template", None):
            # Tokenizer chat templates (Llama/Qwen)
            prompt_text = handle.tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = handle.tok.apply_chat_template(
                [{"role": "user", "content": prompt},
                 {"role": "assistant", "content": completion}],
                tokenize=False,
                add_generation_prompt=False,
            )
        elif use_chat_template and handle.is_processor and hasattr(handle.tok, "apply_chat_template"):
            # Processor chat template (Gemma3)
            prompt_text = self._chat_to_text(handle, [{"role": "user", "content": prompt}], add_generation_prompt=True)
            full_text   = self._chat_to_text(handle, [{"role": "user", "content": prompt},
                                                      {"role": "assistant", "content": completion}], add_generation_prompt=False)
        else:
            prompt_text = prompt
            full_text = prompt + completion

        prompt_ids, _ = self._encode_text(handle, prompt_text)
        full_ids, attn = self._encode_text(handle, full_text)
        prompt_len = prompt_ids.shape[1]
        return full_ids, attn, prompt_len, completion_text
    
    # FIXME: Not Boundary save FML
    # @torch.inference_mode()
    # def calculate_completion_perplexity(
    #     self,
    #     prompt: str,
    #     completion: str,
    #     *,
    #     use_chat_template: bool = True,
    # ) -> Dict[str, Any]:
    #     """
    #     Computes token-level NLL on completion tokens only, returning:
    #       - nll_sum (sum over completion tokens)
    #       - nll_mean_token
    #       - ppl_token = exp(nll_mean_token)
    #       - norms (token/char/byte)
    #     """
    #     handle = self.model_handle
    #     model = handle.model

    #     # Pick a safe device to put inputs on
    #     device = next(model.parameters()).device

    #     input_ids, attention_mask, prompt_len, completion_text =  self._encode_prompt_and_full(
    #         handle, prompt, completion, use_chat_template=use_chat_template
    #     )
    #     input_ids = input_ids.to(device)
    #     attention_mask = attention_mask.to(device)

    #     # Labels: ignore prompt tokens + ignore padding (if any)
    #     labels = input_ids.clone()
    #     labels[:, :prompt_len] = -100
    #     labels = labels.masked_fill(attention_mask == 0, -100)

    #     # Forward -> logits
    #     logits = model(input_ids=input_ids, attention_mask=attention_mask).logits  # [B, T, V]

    #     # Causal LM loss uses next-token prediction: shift logits/labels
    #     shift_logits = logits[:, :-1, :].contiguous()
    #     shift_labels = labels[:, 1:].contiguous()

    #     # Per-token CE (no reduction), ignore_index=-100
    #     vocab = shift_logits.shape[-1]
    #     per_token_loss = F.cross_entropy(
    #         shift_logits.view(-1, vocab),
    #         shift_labels.view(-1),
    #         reduction="none",
    #         ignore_index=-100,
    #     ).view(shift_labels.shape)  # [B, T-1]

    #     scored_mask = shift_labels != -100
    #     n_scored = int(scored_mask.sum().item())

    #     if n_scored == 0:
    #         return {
    #             "nll_sum": float("nan"),
    #             "nll_mean_token": float("nan"),
    #             "ppl_token": float("nan"),
    #             "norms": {
    #                 "n_y_tokens_scored": 0,
    #                 "n_y_chars": len(completion_text),
    #                 "n_y_bytes": len(completion_text.encode("utf-8")),
    #             },
    #         }

    #     nll_sum = float(per_token_loss[scored_mask].sum().item())
    #     nll_mean_token = nll_sum / n_scored
    #     ppl_token = math.exp(nll_mean_token)

    #     # Norms: use raw completion string for chars/bytes (tokenizer-independent)
    #     norms = {
    #         "n_y_tokens_scored": n_scored,                  # tokens that actually contributed to NLL
    #         "n_y_chars": len(completion_text),
    #         "n_y_bytes": len(completion_text.encode("utf-8")),
    #         "prompt_len_tokens": int(prompt_len),
    #         "input_len_tokens": int(input_ids.shape[1]),
    #     }

    #     return {
    #         "nll_sum": nll_sum,
    #         "nll_mean_token": nll_mean_token,
    #         "ppl_token": ppl_token,
    #         "norms": norms,
    #     }

##

def batch_score_last_assistant(
    evaluator,
    conversations,  # list of conversations; each is list[{"role","content",...}]
    *,
    batch_size: int = 8,
    max_length: int | None = None,
    use_chat_template: bool = True,
):
    """
    Returns list of dicts aligned with `conversations`:
      [{"nll_sum", "nll_mean_token", "ppl_token", "norms"}, ...]

    Scientifically robust: scores ONLY the last assistant message by masking tokens
    using *character offsets* from the tokenization of the FULL text (prevents
    prompt/full boundary tokenization mismatch).

    Notes / requirements:
    - Requires a FAST tokenizer to get `offset_mapping`.
    - If the handle is a processor, we use its `.tokenizer` if available; otherwise
      we fall back to the processor itself (may not support offsets).
    - If we cannot determine completion span reliably (e.g., truncation removes it),
      we return NaNs for that example rather than silently wrong numbers.
    """
    handle = evaluator.model_handle
    model = handle.model
    device = next(model.parameters()).device

    # Choose a text tokenizer that supports offsets
    tok = None
    if handle.is_processor:
        # Prefer underlying tokenizer for text-only scoring
        tok = getattr(handle.tok, "tokenizer", None) or getattr(handle.tok, "tokenizer_fast", None) or handle.tok
    else:
        tok = handle.tok

    results = []
    MAX_ns = 0

    def _join_fallback(msg_list):
        # Keep some separators to reduce boundary merges in fallback mode
        parts = []
        for m in msg_list:
            role = (m.get("role") or "").strip()
            content = m.get("content")
            if content is None:
                content = ""
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def to_full_text_and_completion(msgs):
        prompt_msgs, full_msgs, completion_text = evaluator.split_messages_for_scoring(msgs)
        completion_text = completion_text or ""

        if use_chat_template and hasattr(tok, "apply_chat_template"):
            # , enable_thinking=False add here for Qwen3
            prompt_text = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
            full_text = tok.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
        else:
            prompt_text = _join_fallback(prompt_msgs)
            full_text = _join_fallback(full_msgs)

        # Completion span is exactly the suffix added when going from prompt_text -> full_text.
        # This avoids rfind() ambiguity when completion_text appears multiple times.
        comp_start = len(prompt_text)
        comp_end = len(full_text)

        return full_text, completion_text, comp_start, comp_end

    # Mini-batch loop
    for start in range(0, len(conversations), batch_size):
        batch = conversations[start : start + batch_size]

        full_texts = []
        completion_texts = []
        comp_starts = []
        comp_ends = []
        for msgs in batch:
            ftxt, ctxt, cstart, cend = to_full_text_and_completion(msgs)
            full_texts.append(ftxt)
            completion_texts.append(ctxt)
            comp_starts.append(cstart)
            comp_ends.append(cend)

        # Tokenize with offsets (fast tokenizer required)
        try:
            enc = tok(
                full_texts,
                return_tensors="pt",
                padding=True,
                truncation=(max_length is not None),
                max_length=max_length,
                return_offsets_mapping=True,
            )
            offsets = enc["offset_mapping"]  # [B, T, 2] typically on CPU
        except TypeError as e:
            raise TypeError(
                "Tokenizer/processor does not support return_offsets_mapping. "
                "Use a FAST tokenizer (use_fast=True) or access processor.tokenizer."
            ) from e

        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

        # Build labels using offsets: mask everything before completion start (and padding)
        labels = input_ids.clone()

        # If truncation happened, the completion span might be partially/fully gone.
        # We'll detect this by checking if any token offsets overlap the completion span.
        for i in range(input_ids.size(0)):
            cstart = comp_starts[i]
            cend = comp_ends[i]
            ctxt = completion_texts[i] or ""

            # Default: mask everything (=> returns NaNs below)
            labels[i, :] = -100

            # If completion empty, mark as NaN case
            if ctxt is None or len(ctxt) == 0:
                continue

            # Unmask tokens whose offsets overlap the completion span.
            # For completion-only NLL we want to score tokens that correspond to the completion text region.
            any_scored = False
            for t in range(offsets.size(1)):
                if attention_mask[i, t].item() == 0:
                    continue
                s, e = offsets[i, t].tolist()
                # Offset mapping uses [start,end) in characters of the full string.
                # Score token if it overlaps completion span.
                if e > cstart and s < cend:
                    labels[i, t] = input_ids[i, t]
                    any_scored = True

            # If truncation removed completion entirely, keep all -100 so this becomes NaN
            if not any_scored:
                labels[i, :] = -100

        # Also ignore padding
        labels = labels.masked_fill(attention_mask == 0, -100)

        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits  # [B, T, V]

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        vocab = shift_logits.shape[-1]
        # Compute loss in float32 for numerical comparability
        per_token_loss = F.cross_entropy(
            shift_logits.float().view(-1, vocab),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(shift_labels.shape)  # [B, T-1]

        scored_mask = shift_labels != -100
        nll_sums = (per_token_loss * scored_mask).sum(dim=1)   # [B]
        n_scored = scored_mask.sum(dim=1)                      # [B]

        # Package results per row
        for i in range(input_ids.size(0)):
            ns = int(n_scored[i].item())
            ctxt = completion_texts[i] or ""
            if ns == 0:
                results.append(
                    {
                        "nll_sum": float("nan"),
                        "nll_mean_token": float("nan"),
                        "ppl_token": float("nan"),
                        "norms": {
                            "n_y_tokens_scored": 0,
                            "n_y_chars": len(ctxt),
                            "n_y_bytes": len(ctxt.encode("utf-8")),
                            "input_len_tokens": int(input_ids.shape[1]),
                        },
                    }
                )
            else:
                nll_sum = float(nll_sums[i].item())
                nll_mean = nll_sum / ns
                results.append(
                    {
                        "nll_sum": nll_sum,
                        "nll_mean_token": nll_mean,
                        "ppl_token": math.exp(nll_mean),
                        "norms": {
                            "n_y_tokens_scored": ns,
                            "n_y_chars": len(ctxt),
                            "n_y_bytes": len(ctxt.encode("utf-8")),
                            "input_len_tokens": int(input_ids.shape[1]),
                        },
                    }
                )
            if ns > MAX_ns:
                MAX_ns = ns
    print(MAX_ns)

    return results


def flatten_result(res: dict) -> dict:
    """Flatten the output dict so it's CSV-friendly."""
    out = {k: v for k, v in res.items() if k != "norms"}
    norms = res.get("norms", {})
    for nk, nv in norms.items():
        out[f"norm_{nk}"] = nv
    return out


def main():
    data = load_dataset_local("allenai/tulu-3-wildchat-reused-on-policy-8b")

    # FIXME pass enable_thinking=False into apply_chat_template calls for Qwen3?
    MODEL_IDS = [
        # "meta-llama/Llama-2-13b-chat-hf",
        # "meta-llama/Llama-2-7b-chat-hf",
        # "meta-llama/Llama-3.1-8B-Instruct",  
        # "Qwen/Qwen2.5-0.5B-Instruct",
        # "Qwen/Qwen3-8B",   # rerun with enable_thinking=False                    
        # "Qwen/Qwen3-0.6B", # rerun with enable_thinking=False                    
        # "Qwen/Qwen2.5-7B-Instruct", 
        # "google/gemma-2-9b-it",
        # "google/gemma-2-2b-it",
        "google/gemma-2-27b-it",
        # "google/gemma-3-12b-it",
    ]

    # how many dataset rows you want to score
    N = 2400  # change as needed
    batch_rows = 1  # number of dataset examples per batch (each yields 2 convs)
    score_batch_size = 1  # batch_size inside batch_score_last_assistant

    for m in MODEL_IDS:
        evaluator = PerplexityEvaluator(m)
        print(f"Loaded model {m}")

        records = []

        for start in tqdm(range(0, N, batch_rows), desc=f"Scoring rows for {m}"):
            end = min(start + batch_rows, N)
            rows = [data[i] for i in range(start, end)]

            # Build conv list + metadata list in the SAME order
            convs = []
            meta = []
            for r in rows:
                did = r["id"]
                convs.append(r["chosen"])
                meta.append({"dataset_id": did, "which": "chosen"})

                convs.append(r["rejected"])
                meta.append({"dataset_id": did, "which": "rejected"})

            # Score the whole conv batch
            outs = batch_score_last_assistant(
                evaluator,
                convs,
                batch_size=score_batch_size,
            )

            # Merge metadata + outputs into flat records
            for info, res in zip(meta, outs):
                rec = {
                    "model_id": m,
                    **info,
                    **flatten_result(res),
                }
                records.append(rec)

        df = pd.DataFrame.from_records(records)

        # Save per-model file (recommended)
        safe_name = m.replace("/", "__")
        out_path = f"perplexity_{safe_name}.csv"
        df.to_csv(out_path, index=False)
        print(f"Wrote {len(df)} rows to {out_path}")

        del evaluator
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
