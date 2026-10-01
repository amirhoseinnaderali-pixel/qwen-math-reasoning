import argparse
import json
import re
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def extract_gold(answer):
    match = re.search(r"####\s*(-?\d+(?:,\d+)*)", answer)
    if not match:
        raise ValueError(f"No GSM8K final answer found: {answer[:100]}")
    return match.group(1).replace(",", "")


def extract_prediction(text):
    strict = re.search(r"####\s*(-?\d+(?:,\d+)*)", text)
    if strict:
        return strict.group(1).replace(",", ""), True

    # Fallback for generations that solve correctly but omit the GSM8K marker.
    numbers = re.findall(r"(?<![A-Za-z])-?\d+(?:,\d+)*", text)
    if numbers:
        return numbers[-1].replace(",", ""), False
    return None, False


def build_prompt(tokenizer, question):
    messages = [
        {
            "role": "system",
            "content": "You are an expert math tutor. Solve the problem step-by-step and clearly state the final answer using #### <number>.",
        },
        {"role": "user", "content": question},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
    prompt = "<|im_start|>system\n" + messages[0]["content"] + "<|im_end|>\n"
    prompt += "<|im_start|>user\n" + question + "<|im_end|>\n<|im_start|>assistant\n"
    return tokenizer(prompt, return_tensors="pt")["input_ids"]


def load_model(base_model, adapter_path=None):
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval()
    return model, tokenizer


def evaluate(args):
    dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    model, tokenizer = load_model(args.base_model, args.adapter)
    device = next(model.parameters()).device
    correct = 0
    parsed = 0
    strict = 0
    rows = []
    total_generation_seconds = 0.0

    for example in tqdm(dataset, desc=args.mode):
        prompt_ids = build_prompt(tokenizer, example["question"]).to(device)
        started = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(
                prompt_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - started
        total_generation_seconds += elapsed

        generated = tokenizer.decode(
            output_ids[0][prompt_ids.shape[-1]:],
            skip_special_tokens=True,
        ).strip()
        predicted, strict_format = extract_prediction(generated)
        gold = extract_gold(example["answer"])
        is_parsed = predicted is not None
        is_correct = is_parsed and predicted == gold
        correct += int(is_correct)
        parsed += int(is_parsed)
        strict += int(strict_format)

        rows.append(
            {
                "question": example["question"],
                "gold": gold,
                "prediction": predicted,
                "correct": is_correct,
                "parsed": is_parsed,
                "strict_format": strict_format,
                "generation_seconds": elapsed,
                "output": generated,
            }
        )

    n = len(rows)
    result = {
        "mode": args.mode,
        "base_model": args.base_model,
        "adapter": args.adapter,
        "dataset": args.dataset,
        "split": args.split,
        "examples": n,
        "exact_match_accuracy": correct / n if n else 0.0,
        "parse_rate": parsed / n if n else 0.0,
        "strict_hash4_rate": strict / n if n else 0.0,
        "mean_generation_seconds": total_generation_seconds / n if n else 0.0,
        "rows": rows,
    }

    output_path = Path(args.output or f"results/{args.mode}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["base", "adapter"], required=True)
    parser.add_argument("--base-model", default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--dataset", default="openai/gsm8k")
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.mode == "adapter" and not args.adapter:
        parser.error("--adapter is required for --mode adapter")
    evaluate(args)