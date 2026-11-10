#!/usr/bin/env python3
"""Run the samplers on a real small model, and test how repeatable greedy decoding is.

Needs torch and transformers. Writes results/sampling.json and results/determinism.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import samplers

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
PROMPTS = [
    "Explain what a load balancer does.",
    "What causes a memory leak?",
    "Describe how DNS resolution works.",
    "Write one sentence about the ocean.",
    "What is a good name for a pet cat?",
    "Suggest a title for a story about a lighthouse.",
    "Why do databases use indexes?",
    "What should I check when a website is slow?",
    "Describe the taste of coffee.",
    "Give me an idea for a weekend project.",
    "What is the capital of France, and why is it famous?",
    "Explain recursion to a beginner.",
    "How does a vaccine work?",
    "Write a short greeting for a new colleague.",
    "What is the difference between a process and a thread?",
    "Name three uses for a paperclip.",
    "Why is the sky blue?",
    "What makes a good incident report?",
    "Describe a quiet morning in a small town.",
    "How do you make a cup of tea?",
]
SETTINGS = {
    "T=0.7": {"temperature": 0.7},
    "T=1.0": {"temperature": 1.0},
    "T=1.5": {"temperature": 1.5},
    "T=1.5, top-p 0.9": {"temperature": 1.5, "top_p": 0.9},
    "T=1.5, min-p 0.1": {"temperature": 1.5, "min_p": 0.1},
    "T=1.0, top-k 40": {"temperature": 1.0, "top_k": 40},
    "T=1.0, top-p 0.9": {"temperature": 1.0, "top_p": 0.9},
    "T=1.0, min-p 0.1": {"temperature": 1.0, "min_p": 0.1},
}
TAIL = 0.01  # a token the untempered model itself gave under 1%


def encode(tokenizer, prompt: str) -> list[int]:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=False
    )


@torch.inference_mode()
def sample_run(model, tokenizer, ids, setting, length, rng):
    tokens, tail, past, inputs = [], 0, None, torch.tensor([ids])
    for _ in range(length):
        out = model(input_ids=inputs, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1].numpy().astype(np.float64)
        probs = samplers.filtered(logits, **setting)
        token = int(rng.choice(probs.size, p=probs))
        tail += samplers.softmax(logits)[token] < TAIL
        tokens.append(token)
        if token == tokenizer.eos_token_id:
            break
        inputs = torch.tensor([[token]])
    return tokens, tail


def sampling_experiment(model, tokenizer, samples, length, seed):
    rows = []
    for label, setting in SETTINGS.items():
        rng = np.random.default_rng(seed)
        tail, total, distinct, repeats = 0, 0, [], []
        for prompt in PROMPTS:
            ids = encode(tokenizer, prompt)
            outputs = []
            for _ in range(samples):
                tokens, t = sample_run(model, tokenizer, ids, setting, length, rng)
                tail += t
                total += len(tokens)
                outputs.append(tuple(tokens))
                grams = list(zip(tokens, tokens[1:], tokens[2:], strict=False))
                repeats.append(1 - len(set(grams)) / len(grams) if grams else 0)
            distinct.append(len({o[:16] for o in outputs}) / samples)
        rows.append(
            {
                "setting": label,
                "tail_token_rate": round(tail / total, 4),
                "distinct_openings": round(statistics.mean(distinct), 3),
                "repeated_trigram_rate": round(statistics.mean(repeats), 4),
                "tokens": total,
            }
        )
        print(rows[-1], flush=True)
    return rows


@torch.inference_mode()
def greedy_tokens(model, tokenizer, ids, length, device, batch_mates=()):
    """Greedy decode one prompt, optionally padded into a batch with other prompts."""
    sequences = [list(ids), *[list(m) for m in batch_mates]]
    out = []
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    for _ in range(length):
        width = max(map(len, sequences))
        batch = torch.tensor([[pad] * (width - len(s)) + s for s in sequences], device=device)
        mask = torch.tensor(
            [[0] * (width - len(s)) + [1] * len(s) for s in sequences], device=device
        )
        position_ids = (mask.cumsum(1) - 1).clamp(min=0)
        logits = model(input_ids=batch, attention_mask=mask, position_ids=position_ids).logits[
            :, -1
        ]
        next_tokens = logits.argmax(-1).tolist()
        out.append(next_tokens[0])
        if next_tokens[0] == tokenizer.eos_token_id:
            break
        sequences = [s + [t] for s, t in zip(sequences, next_tokens, strict=True)]
    return out


def first_difference(a, b):
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def determinism_experiment(model, tokenizer, length):
    encoded = [encode(tokenizer, p) for p in PROMPTS]
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    report = {"prompts": len(PROMPTS), "length": length, "devices": {}}
    alone_by_device = {}
    for device in devices:
        model.to(device)
        alone = [greedy_tokens(model, tokenizer, e, length, device) for e in encoded]
        repeat = [greedy_tokens(model, tokenizer, e, length, device) for e in encoded]
        padded = [
            greedy_tokens(
                model, tokenizer, e, length, device, encoded[(i + 1) % 20 : (i + 1) % 20 + 7]
            )
            for i, e in enumerate(encoded)
        ]
        alone_by_device[device] = alone
        diverged = [first_difference(a, b) for a, b in zip(alone, padded, strict=True)]
        report["devices"][device] = {
            "repeat_runs_identical": sum(a == b for a, b in zip(alone, repeat, strict=True)),
            "alone_vs_batched_identical": sum(d is None for d in diverged),
            "first_divergence_positions": [d for d in diverged if d is not None],
        }
        print(device, report["devices"][device], flush=True)
    if len(devices) == 2:
        diverged = [
            first_difference(a, b)
            for a, b in zip(alone_by_device["cpu"], alone_by_device["mps"], strict=True)
        ]
        report["cpu_vs_mps_identical"] = sum(d is None for d in diverged)
        report["cpu_vs_mps_first_divergence"] = [d for d in diverged if d is not None]
    model.to("cpu")
    return report


@torch.inference_mode()
def last_logprobs(model, sequences, pad, device):
    width = max(map(len, sequences))
    batch = torch.tensor([[pad] * (width - len(s)) + s for s in sequences], device=device)
    mask = torch.tensor([[0] * (width - len(s)) + [1] * len(s) for s in sequences], device=device)
    position_ids = (mask.cumsum(1) - 1).clamp(min=0)
    logits = model(input_ids=batch, attention_mask=mask, position_ids=position_ids).logits[0, -1]
    return torch.log_softmax(logits.float(), -1).cpu()


def top_gap(logprobs, order) -> float:
    return float(logprobs[order[0]] - logprobs[order[1]])


@torch.inference_mode()
def margin_experiment(model, tokenizer, length):
    """How close does greedy decoding come to a tie, against the size of the numeric noise?"""
    encoded = [encode(tokenizer, p) for p in PROMPTS]
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    gaps, batch_noise, device_noise = [], [], []
    use_mps = torch.backends.mps.is_available()
    for i, ids in enumerate(encoded):
        tokens, past, inputs = list(ids), None, torch.tensor([ids])
        for _ in range(length):
            out = model(input_ids=inputs, past_key_values=past, use_cache=True)
            past = out.past_key_values
            top = torch.log_softmax(out.logits[0, -1], -1).topk(2)
            gaps.append(float(top.values[0] - top.values[1]))
            token = int(top.indices[0])
            tokens.append(token)
            if token == tokenizer.eos_token_id:
                break
            inputs = torch.tensor([[token]])
        mates = [list(m) for m in encoded[(i + 1) % 20 : (i + 1) % 20 + 7]]
        alone = last_logprobs(model, [tokens], pad, "cpu")
        batched = last_logprobs(model, [tokens, *mates], pad, "cpu")
        order = alone.topk(2).indices
        batch_noise.append(abs(top_gap(alone, order) - top_gap(batched, order)))
        if use_mps:
            model.to("mps")
            other = last_logprobs(model, [tokens], pad, "mps")
            model.to("cpu")
            device_noise.append(abs(top_gap(alone, order) - top_gap(other, order)))
    ordered = sorted(gaps)
    return {
        "steps": len(gaps),
        "median_top2_gap": ordered[len(ordered) // 2],
        "smallest_top2_gap": ordered[0],
        "steps_with_gap_under_0.01": sum(g < 0.01 for g in gaps),
        "batch_noise_median": statistics.median(batch_noise),
        "batch_noise_max": max(batch_noise),
        "device_noise_median": statistics.median(device_noise) if device_noise else None,
        "device_noise_max": max(device_noise) if device_noise else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--margins-only", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    Path("results").mkdir(exist_ok=True)
    meta = {"model": MODEL, "seed": args.seed, "samples": args.samples, "length": args.length}
    if args.margins_only:
        margins = {"model": MODEL, **margin_experiment(model, tokenizer, 128)}
        Path("results/margins.json").write_text(json.dumps(margins, indent=2) + "\n")
        print(margins)
        return
    determinism = determinism_experiment(model, tokenizer, 128)
    Path("results/determinism.json").write_text(
        json.dumps({**meta, **determinism}, indent=2) + "\n"
    )
    rows = sampling_experiment(model, tokenizer, args.samples, args.length, args.seed)
    Path("results/sampling.json").write_text(json.dumps({**meta, "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
