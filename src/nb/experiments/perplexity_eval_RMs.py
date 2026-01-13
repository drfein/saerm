"""
Evaluate reward models on the same dataset as perplexity.py.

Computes reward scores for each (prompt, completion) pair from 
allenai/tulu-3-wildchat-reused-on-policy-8b dataset.
"""

from __future__ import annotations

import pandas as pd
from datasets import load_dataset
from typing import List, Dict, Any
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def format_conversation(tokenizer: Any, messages: List[Dict[str, str]]) -> str:
    """Format a conversation for the reward model using chat template.
    
    Args:
        tokenizer: HuggingFace tokenizer with chat template
        messages: List of message dicts with 'role' and 'content'
        
    Returns:
        Formatted conversation string
    """
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        # formatted = tokenizer.apply_chat_template(
        #     messages,
        #     tokenize=False,
        #     add_generation_prompt=False,
        # )
        try:
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
        except TypeError:
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        # Remove BOS token - it will be added back during tokenization
        if tokenizer.bos_token and formatted.startswith(tokenizer.bos_token):
            formatted = formatted[len(tokenizer.bos_token):]
        return formatted
    else:
        # Fallback: simple concatenation
        return "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)


def get_rewards_batch(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: List[str],
    batch_size: int = 8,
    device: str = "cuda",
    max_length: int = 20048,
) -> List[float]:
    """Compute reward scores for a list of texts.
    
    Args:
        model: Reward model
        tokenizer: Tokenizer
        texts: List of formatted conversation texts
        batch_size: Batch size for inference
        device: Device to use
        max_length: Maximum sequence length
        
    Returns:
        List of reward scores
    """
    model.eval()
    all_scores = []
    
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="Computing rewards"):
            batch_texts = texts[start : start + batch_size]
            
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            
            # Extract scores - handle different model output formats
            if hasattr(outputs, "logits"):
                scores = outputs.logits.squeeze(-1)
            elif hasattr(outputs, "score"):
                scores = outputs.score
            else:
                scores = outputs[0].squeeze(-1) if isinstance(outputs, tuple) else outputs.squeeze(-1)

            assert scores.ndim == 1, f"Expected scalar reward per example, got shape {outputs.logits.shape}"
            
            all_scores.extend(scores.cpu().tolist())
    
    return all_scores


def main():
    # Configuration
    MODEL_IDs = [
        # "Skywork/Skywork-Reward-V2-Llama-3.1-8B",
        # "Skywork/Skywork-Reward-V2-Qwen3-8B", 
        # "Skywork/Skywork-Reward-V2-Qwen3-0.6B", 
        "allenai/Llama-3.1-8B-Instruct-RM-RB2",
    ]
    N = 2400  # Same as perplexity.py
    BATCH_SIZE = 4
    MAX_LENGTH = 200048
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load dataset (same as perplexity.py)
    print("Loading dataset: allenai/tulu-3-wildchat-reused-on-policy-8b")
    data = load_dataset("allenai/tulu-3-wildchat-reused-on-policy-8b", split="train")
    
    # Load reward model
    for MODEL in MODEL_IDs:
        print(f"Loading reward model: {MODEL}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # model.config.pad_token_id = tokenizer.pad_token_id
        # if getattr(model.generation_config, "pad_token_id", None) is None:
        #     model.generation_config.pad_token_id = tokenizer.pad_token_id
        
        model.config.pad_token_id = tokenizer.pad_token_id
        # only touch generation_config if it exists
        if getattr(model, "generation_config", None) is not None:
            if getattr(model.generation_config, "pad_token_id", None) is None:
                model.generation_config.pad_token_id = tokenizer.pad_token_id
        
        # Prepare texts
        print(f"Formatting {N} examples...")
        records = []
        texts = []
        meta = []
        
        for i in tqdm(range(N), desc="Preparing data"):
            row = data[i]
            dataset_id = row["id"]
            
            # Format chosen
            chosen_text = format_conversation(tokenizer, row["chosen"])
            texts.append(chosen_text)
            meta.append({"dataset_id": dataset_id, "which": "chosen"})
            
            # Format rejected
            rejected_text = format_conversation(tokenizer, row["rejected"])
            texts.append(rejected_text)
            meta.append({"dataset_id": dataset_id, "which": "rejected"})
        
        # Compute rewards
        print("Computing rewards...")
        rewards = get_rewards_batch(
            model, tokenizer, texts,
            batch_size=BATCH_SIZE,
            device=DEVICE,
            max_length=MAX_LENGTH,
        )
        
        # Build records
        for m, reward in zip(meta, rewards):
            records.append({
                "model_id": MODEL,
                "dataset_id": m["dataset_id"],
                "which": m["which"],
                "reward": reward,
            })
        
        # Save results
        df = pd.DataFrame.from_records(records)
        safe_name = MODEL.replace("/", "__")
        out_path = f"rewards_{safe_name}.csv"
        df.to_csv(out_path, index=False)
        print(f"Wrote {len(df)} rows to {out_path}")
        
        # Summary
        print("\nSummary:")
        print(df.groupby("which")["reward"].describe())
        
        # Accuracy
        chosen = df[df["which"] == "chosen"].set_index("dataset_id")["reward"]
        rejected = df[df["which"] == "rejected"].set_index("dataset_id")["reward"]
        accuracy = (chosen > rejected).mean()
        print(f"\nAccuracy (chosen > rejected): {accuracy:.2%}")


if __name__ == "__main__":
    main()