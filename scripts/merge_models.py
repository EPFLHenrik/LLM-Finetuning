#!/usr/bin/env python3
"""Merge two model checkpoints using LERP or SLERP.

Both models must share the same architecture (same base model, same parameter
shapes). The merged weights are saved with the tokenizer from model-a.

Methods:
  lerp  — linear interpolation: (1-alpha)*A + alpha*B  (default, fast, reliable)
  slerp — spherical linear interpolation per tensor     (better direction preservation)

Usage:
    python scripts/merge_models.py \
        --model-a models/full_sft \
        --model-b models/full_math_grpo \
        --output  models/merged_sft_grpo \
        --alpha   0.5 \
        --method  lerp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _lerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    return ((1.0 - t) * a.float() + t * b.float()).to(a.dtype)


def _slerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    a_f = a.flatten().float()
    b_f = b.flatten().float()
    norm_a = a_f.norm()
    norm_b = b_f.norm()
    if norm_a < 1e-8 or norm_b < 1e-8:
        return _lerp(a, b, t)
    cos_theta = torch.clamp(torch.dot(a_f / norm_a, b_f / norm_b), -1.0, 1.0)
    theta = torch.acos(cos_theta)
    if theta.abs() < 1e-6:
        return _lerp(a, b, t)
    sin_theta = torch.sin(theta)
    w_a = torch.sin((1.0 - t) * theta) / sin_theta
    w_b = torch.sin(t * theta) / sin_theta
    merged_flat = w_a * a_f + w_b * b_f
    # Interpolate magnitude linearly
    merged_norm = (1.0 - t) * norm_a + t * norm_b
    merged_flat = merged_flat / merged_flat.norm() * merged_norm
    return merged_flat.reshape(a.shape).to(a.dtype)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge two model checkpoints with LERP or SLERP.")
    parser.add_argument("--model-a", required=True, type=Path, help="First model folder (tokenizer is copied from here)")
    parser.add_argument("--model-b", required=True, type=Path, help="Second model folder")
    parser.add_argument("--output", required=True, type=Path, help="Output folder for merged model")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Interpolation weight: 0 = all model-a, 1 = all model-b (default: 0.5)")
    parser.add_argument("--method", choices=("lerp", "slerp"), default="lerp",
                        help="Interpolation method (default: lerp)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to load weights onto (default: cpu — saves GPU memory)")
    args = parser.parse_args(argv)

    if not args.model_a.exists():
        raise SystemExit(f"model-a not found: {args.model_a}")
    if not args.model_b.exists():
        raise SystemExit(f"model-b not found: {args.model_b}")
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}. Delete it first.")

    interp_fn = _slerp if args.method == "slerp" else _lerp

    print(f"Loading model A from {args.model_a} ...")
    model_a = AutoModelForCausalLM.from_pretrained(
        str(args.model_a), torch_dtype=torch.float32, trust_remote_code=True,
    ).to(args.device)

    print(f"Loading model B from {args.model_b} ...")
    model_b = AutoModelForCausalLM.from_pretrained(
        str(args.model_b), torch_dtype=torch.float32, trust_remote_code=True,
    ).to(args.device)

    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()

    keys_a = set(sd_a.keys())
    keys_b = set(sd_b.keys())
    if keys_a != keys_b:
        only_a = keys_a - keys_b
        only_b = keys_b - keys_a
        raise SystemExit(
            f"State dict mismatch:\n  only in A: {only_a}\n  only in B: {only_b}"
        )

    print(f"Merging {len(sd_a)} tensors with {args.method} at alpha={args.alpha} ...")
    merged_sd = {}
    for key in sd_a:
        pa = sd_a[key]
        pb = sd_b[key]
        if pa.shape != pb.shape:
            raise SystemExit(f"Shape mismatch for {key}: {pa.shape} vs {pb.shape}")
        if pa.is_floating_point():
            merged_sd[key] = interp_fn(pa, pb, args.alpha)
        else:
            # Non-float params (e.g. position ids): keep from model-a
            merged_sd[key] = pa.clone()

    # Write merged weights into model-a's structure and save
    model_a.load_state_dict(merged_sd)
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to {args.output} ...")
    model_a.save_pretrained(str(args.output), safe_serialization=True)
    if getattr(model_a, "generation_config", None) is not None:
        model_a.generation_config.save_pretrained(str(args.output))

    print("Saving tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_a), trust_remote_code=True)
    tokenizer.save_pretrained(str(args.output))

    print(f"\nDone. Merged model written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
