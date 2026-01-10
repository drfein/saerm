#!/usr/bin/env python3
"""
Generate length bias dataset from GSM8K.

This script:
1. Loads GSM8K questions
2. Generates correct and incorrect solutions using an LLM
3. Creates verbose versions of correct answers
4. Saves to the expected JSON format for LengthBiasDataset

Usage:
    python -m src.nb.datasets.generate_length_data \
        --output data/gsm8k_soln.json \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --n-questions 1000
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

logger = logging.getLogger(__name__)


def parse_gold_number(gold_answer: str) -> Optional[float]:
    """Parse gold answer to float, handling commas."""
    try:
        return float(gold_answer.replace(",", ""))
    except Exception:
        return None


def extract_final_answer(solution: str) -> Optional[str]:
    """Extract the final numerical answer from a solution."""
    # Look for #### format (GSM8K standard)
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", solution)
    if match:
        return match.group(1).replace(",", "")
    
    # Look for "answer is X" format
    match = re.search(r"(?:answer|result|solution)\s+(?:is|=|:)\s*(-?[\d,]+(?:\.\d+)?)", solution, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "")
    
    # Look for final number in the solution
    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", solution)
    if numbers:
        return numbers[-1].replace(",", "")
    
    return None


def check_answer(solution: str, gold_answer: str) -> bool:
    """Check if the solution contains the correct answer."""
    extracted = extract_final_answer(solution)
    if extracted is None:
        return False
    
    # Normalize both answers
    try:
        extracted_num = float(extracted)
        gold_num = float(gold_answer.replace(",", ""))
        return abs(extracted_num - gold_num) < 1e-6
    except (ValueError, TypeError):
        return extracted == gold_answer.replace(",", "")


def generate_solution(
    generator,
    tokenizer,
    question: str,
    make_incorrect: bool = False,
    max_new_tokens: int = 512,
    use_vllm: bool = False,
) -> str:
    """Generate a solution for a math question.

    If make_incorrect=True, we sample with a normal solve prompt and rely on
    stochasticity to produce wrong answers (no forced perturbation).
    """
    prompt = f"""Solve this math problem step by step:

{question}

Show your work and end with #### followed by just the final numerical answer.
Your answer may contain a mistake; do not apologize or mention uncertainty."""

    messages = [{"role": "user", "content": prompt}]

    if use_vllm:
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=max_new_tokens,
        )
        # Build a prompt string using the chat template; vLLM expects a string prompt
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        outputs = generator.generate(prompt_text, sampling_params)
        # vLLM returns list of RequestOutput; take first candidate
        response = outputs[0].outputs[0].text
        return response.strip()

    # HF generate path
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(generator.device)
    
    with torch.no_grad():
        outputs = generator.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated = outputs[0][inputs.shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)
    
    return response.strip()


def generate_batch_vllm(
    generator,
    tokenizer,
    prompt_messages: List[List[Dict[str, str]]],
    max_new_tokens: int,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> List[str]:
    """Generate a batch of prompts with vLLM (expects list of chat messages)."""
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
    )
    prompt_texts = [
        tokenizer.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            tokenize=False,
        )
        for msgs in prompt_messages
    ]
    outputs = generator.generate(prompt_texts, sampling_params)
    return [out.outputs[0].text.strip() for out in outputs]


def make_verbose(
    generator,
    tokenizer,
    solution: str,
    max_new_tokens: int = 1024,
    use_vllm: bool = False,
) -> str:
    """Rewrite a solution in a verbose way."""
    prompt = f"""Rewrite the following solution in a much more verbose way. Add more explanation, detail each step thoroughly, and include additional context. Make it at least 2-3 times longer while keeping the same answer:

Original solution:
{solution}

Verbose rewrite:"""

    messages = [{"role": "user", "content": prompt}]

    if use_vllm:
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=max_new_tokens,
        )
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        outputs = generator.generate(prompt_text, sampling_params)
        response = outputs[0].outputs[0].text
        return response.strip()

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(generator.device)
    
    with torch.no_grad():
        outputs = generator.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated = outputs[0][inputs.shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)
    
    return response.strip()


def generate_length_dataset(
    model_path: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    output_path: str = "data/gsm8k_soln.json",
    n_questions: int = 1000,
    max_attempts: int = 3,
    device: str = "cuda",
    trust_remote_code: bool = True,
    use_vllm: bool = False,
    vllm_batch_size: int = 16,
) -> Dict[str, Any]:
    """Generate the length bias dataset.
    
    Args:
        model_path: Path or HuggingFace ID of the generation model
        output_path: Output JSON file path
        n_questions: Number of questions to process
        max_attempts: Max attempts to generate correct/incorrect pairs
        device: Device for inference
        trust_remote_code: Whether to trust remote code
        use_vllm: Use vLLM for faster generation
        vllm_batch_size: Batch size for vLLM generation
        
    Returns:
        The generated dataset dictionary
    """
    logger.info("Loading model: %s (use_vllm=%s)", model_path, use_vllm)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)

    if use_vllm:
        try:
            from vllm import LLM
        except ImportError as exc:
            raise ImportError("vLLM is not installed. pip install vllm") from exc

        # vLLM handles its own tokenizer internally; pass model path only.
        generator = LLM(
            model=model_path,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=1,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=trust_remote_code,
        )
        generator = model
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    
    logger.info("Loading GSM8K dataset...")
    gsm8k = load_dataset("openai/gsm8k", "main", split="train")
    
    # Resume if output exists
    questions: List[Dict[str, Any]] = []
    existing_indices: set[int] = set()
    out_path = Path(output_path)
    if out_path.exists():
        try:
            with open(out_path) as f:
                data_existing = json.load(f)
            questions = data_existing.get("questions", [])
            existing_indices = {q.get("question_idx") for q in questions if "question_idx" in q}
            logger.info("Resuming from existing file with %d questions", len(questions))
        except Exception as exc:
            logger.warning("Could not load existing output for resume: %s", exc)
            questions = []
            existing_indices = set()

    # Limit to n_questions (if <=0, use all)
    if n_questions is not None and n_questions > 0:
        gsm8k = gsm8k.select(range(min(n_questions, len(gsm8k))))
    
    logger.info("Generating solutions for %d questions...", len(gsm8k))
    
    if not use_vllm:
        for idx, item in enumerate(tqdm(gsm8k, desc="Generating")):
            if idx in existing_indices:
                continue
            question_text = item["question"]
            gold_answer = item["answer"].split("####")[-1].strip()
            
            logger.debug("Processing question %d: %s", idx, question_text[:50])
            
            # Generate correct solution
            correct_solution = None
            for attempt in range(max_attempts):
                solution = generate_solution(generator, tokenizer, question_text, make_incorrect=False, use_vllm=False)
                if check_answer(solution, gold_answer):
                    correct_solution = solution
                    break
                logger.debug("Attempt %d: incorrect answer, retrying...", attempt + 1)
            
            if correct_solution is None:
                logger.warning("Could not generate correct solution for question %d, skipping", idx)
                continue
            
        # Generate incorrect solution: try to sample a wrong answer (no forced errors)
            incorrect_solution = None
            for attempt in range(max_attempts):
                solution = generate_solution(generator, tokenizer, question_text, make_incorrect=True, use_vllm=False)
                if not check_answer(solution, gold_answer):
                    incorrect_solution = solution
                    break
                logger.debug("Attempt %d: accidentally correct, retrying...", attempt + 1)
            
            if incorrect_solution is None:
                logger.warning("Could not sample incorrect solution for question %d, skipping", idx)
                continue
            
            logger.debug("Generating verbose version...")
            correct_verbose = make_verbose(generator, tokenizer, correct_solution, use_vllm=False)
            
            questions.append({
                "question_idx": idx,
                "question": question_text,
                "gold_answer": gold_answer,
                "solutions": [
                    {"response": correct_solution, "is_correct": True, "variant": "correct"},
                    {"response": incorrect_solution, "is_correct": False, "variant": "incorrect"},
                    {"response": correct_verbose, "is_correct": True, "variant": "correct_verbose"},
                ],
            })
            
            if (idx + 1) % 50 == 0:
                logger.info("Processed %d questions, saving checkpoint...", idx + 1)
                save_dataset(questions, output_path)
    else:
        batch_size = vllm_batch_size
        total = len(gsm8k)
        remaining_indices = [i for i in range(total) if i not in existing_indices]
        num_batches = ceil(len(remaining_indices) / batch_size) if remaining_indices else 0
        for b in tqdm(range(num_batches), desc="Generating (vLLM)"):
            idx_slice = remaining_indices[b * batch_size : (b + 1) * batch_size]
            if not idx_slice:
                continue
            batch = gsm8k.select(idx_slice)
            batch_indices = idx_slice
            questions_batch = [item["question"] for item in batch]
            gold_batch = [item["answer"].split("####")[-1].strip() for item in batch]

            # Correct solutions with retries
            pending = list(range(len(batch)))
            correct_solutions: Dict[int, str] = {}
            for attempt in range(max_attempts):
                if not pending:
                    break
                prompt_msgs = [
                    [
                        {
                            "role": "user",
                            "content": (
                                "Solve this math problem step by step:\n\n"
                                f"{questions_batch[i]}\n\n"
                                "Show your work and end with #### followed by just the final numerical answer."
                            ),
                        }
                    ]
                    for i in pending
                ]
                outs = generate_batch_vllm(generator, tokenizer, prompt_msgs, max_new_tokens=512)
                still = []
                for idx_local, out in zip(pending, outs):
                    if check_answer(out, gold_batch[idx_local]):
                        correct_solutions[idx_local] = out
                    else:
                        still.append(idx_local)
                pending = still

            valid_local = [i for i in range(len(batch)) if i in correct_solutions]
            if not valid_local:
                continue

            # Incorrect solutions: sample answers and keep those that are wrong
            incorrect_solutions: Dict[int, str] = {}
            pending_wrong = valid_local.copy()
            for attempt in range(max_attempts):
                if not pending_wrong:
                    break
                prompt_msgs = [
                    [
                        {
                            "role": "user",
                            "content": (
                                "Solve this math problem step by step:\n\n"
                                f"{questions_batch[i]}\n\n"
                                "Show your work and end with #### followed by just the final numerical answer."
                            ),
                        }
                    ]
                    for i in pending_wrong
                ]
                outs = generate_batch_vllm(generator, tokenizer, prompt_msgs, max_new_tokens=512)
                still_pending = []
                for i_local, out in zip(pending_wrong, outs):
                    if not check_answer(out, gold_batch[i_local]):
                        incorrect_solutions[i_local] = out
                    else:
                        still_pending.append(i_local)
                pending_wrong = still_pending

            # Verbose rewrites batch
            verbose_solutions: Dict[int, str] = {}
            prompt_msgs = [
                [
                    {
                        "role": "user",
                        "content": (
                            "Rewrite the following solution in a much more verbose way. "
                            "Add more explanation, detail each step thoroughly, and include additional context. "
                            "Make it at least 2-3 times longer while keeping the same answer:\n\n"
                            f"Original solution:\n{correct_solutions[i]}\n\nVerbose rewrite:"
                        ),
                    }
                ]
                for i in valid_local
            ]
            outs = generate_batch_vllm(generator, tokenizer, prompt_msgs, max_new_tokens=1024)
            for i_local, out in zip(valid_local, outs):
                verbose_solutions[i_local] = out

            # Assemble
            for local_idx in valid_local:
                global_idx = batch_indices[local_idx]
                correct_solution = correct_solutions[local_idx]
                incorrect_solution = incorrect_solutions.get(local_idx)
                # If we failed to get an incorrect, skip this example
                if not incorrect_solution or check_answer(incorrect_solution, gold_batch[local_idx]):
                    continue

                correct_verbose = verbose_solutions.get(local_idx, correct_solution)

                questions.append(
                    {
                        "question_idx": global_idx,
                        "question": questions_batch[local_idx],
                        "gold_answer": gold_batch[local_idx],
                        "solutions": [
                            {"response": correct_solution, "is_correct": True, "variant": "correct"},
                            {"response": incorrect_solution, "is_correct": False, "variant": "incorrect"},
                            {"response": correct_verbose, "is_correct": True, "variant": "correct_verbose"},
                        ],
                    }
                )

            if (b + 1) % 5 == 0:
                logger.info("Processed %d/%d questions, saving checkpoint...", len(questions), total)
                save_dataset(questions, output_path)
    
    logger.info("Generated %d valid question-answer pairs", len(questions))
    
    # Final save
    save_dataset(questions, output_path)
    
    return {"questions": questions}


def save_dataset(questions: List[Dict], output_path: str) -> None:
    """Save the dataset to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump({"questions": questions}, f, indent=2)
    
    logger.info("Saved %d questions to %s", len(questions), output_path)


def main():
    parser = argparse.ArgumentParser(description="Generate length bias dataset from GSM8K")
    parser.add_argument(
        "--output",
        type=str,
        default="data/gsm8k_soln.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Generation model path or HuggingFace ID",
    )
    parser.add_argument(
        "--n-questions",
        type=int,
        default=1000,
        help="Number of questions to process",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Max attempts to generate correct/incorrect pairs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference",
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="Use vLLM for generation (requires GPU and vllm installed)",
    )
    parser.add_argument(
        "--vllm-batch-size",
        type=int,
        default=16,
        help="Batch size for vLLM generation",
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    generate_length_dataset(
        model_path=args.model,
        output_path=args.output,
        n_questions=args.n_questions,
        max_attempts=args.max_attempts,
        device=args.device,
        use_vllm=args.use_vllm,
        vllm_batch_size=args.vllm_batch_size,
    )


if __name__ == "__main__":
    main()

