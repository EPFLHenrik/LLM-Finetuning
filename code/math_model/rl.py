"""Reinforcement learning with verifiable rewards (RLVR) for Qwen3-1.7B math.

Thin Python wrapper around ms-swift's `swift rlhf` CLI. Both RL phases of the
shipped pipeline use this entry point; they differ only in `--model` (the
initialisation) and `--dataset` (the learnability-filtered pool).

Two policy-optimisation objectives are supported via `--loss`:

  - gspo  (default; sequence-level importance sampling)
          The shipped model uses this. Implemented by passing
          `--loss_type grpo --importance_sampling_level sequence` to swift,
          which is the TRL flag combination for GSPO.

  - cispo (token-level importance sampling with detached IS-weight clipping)
          Same family as GRPO; used in the original 55% CI baseline run.

Usage (from the repo root):

    # Phase 1: from the SFT-merged checkpoint, base-filtered learnability pool.
    python -m code.math_model.rl \\
        --model    /scratch/<user>/ckpt/qwen3_1.7b_sft600_merged \\
        --dataset  /scratch/<user>/rl_data/deepscaler_filtered.jsonl \\
        --output   /scratch/<user>/ckpt/math_rl_phase1 \\
        --steps 500

    # Phase 2: from the phase-1-merged checkpoint, policy-refiltered pool.
    python -m code.math_model.rl \\
        --model    /scratch/<user>/ckpt/math_rl_phase1_merged \\
        --dataset  /scratch/<user>/rl_data/deepscaler_filtered_phase1.jsonl \\
        --output   /scratch/<user>/ckpt/math_rl_phase2 \\
        --steps 800

After each phase, merge the LoRA into the base before continuing:

    python -m code.math_model.huggingface.push_checkpoint \\
        /scratch/<user>/ckpt/math_rl_phase1/checkpoint-500 \\
        --local-out /scratch/<user>/ckpt/math_rl_phase1_merged \\
        --skip-vllm-check
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


# Phase schedule (mirrors the 4K -> 8K -> 16K curriculum from DeepScaleR).
# The shipped model uses PHASE=8k for both RL phases.
PHASE_DEFAULTS = {
    "4k":  {"max_completion": 4096,  "num_gen": 8, "default_steps": 500},
    "8k":  {"max_completion": 8192,  "num_gen": 4, "default_steps": 500},  # shipped
    "16k": {"max_completion": 16384, "num_gen": 4, "default_steps": 300},
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="HF id or local path of the initialisation model. "
                         "Phase 1: the SFT-merged checkpoint. "
                         "Phase 2: the phase-1-merged checkpoint.")
    ap.add_argument("--dataset", required=True,
                    help="JSONL produced by preprocessing.rl.filter_learnability "
                         "+ preprocessing.rl.add_system_prompt.")
    ap.add_argument("--output", required=True,
                    help="Directory for LoRA checkpoints (will be created).")

    # RL objective
    ap.add_argument("--loss", default="gspo", choices=["gspo", "cispo"],
                    help="Policy-optimisation objective. gspo (default, shipped) "
                         "uses sequence-level IS; cispo uses token-level IS "
                         "with detached-weight clipping (matches the original "
                         "55% CI baseline).")
    ap.add_argument("--eps-high", type=float, default=5.0,
                    help="CISPO clip threshold. Ignored when --loss=gspo.")

    # Phase schedule
    ap.add_argument("--phase", default="8k", choices=list(PHASE_DEFAULTS),
                    help="Training context budget (default 8k, used by shipped).")
    ap.add_argument("--num-generations", type=int, default=None,
                    help="Group size N. Default depends on --phase "
                         "(8 at 4K, 4 at 8K/16K to fit a 40 GB A100).")

    # Training hyperparameters
    ap.add_argument("--steps", type=int, default=None,
                    help="Total optimiser steps. Default depends on --phase.")
    ap.add_argument("--lr", type=float, default=1e-6,
                    help="Learning rate. 1e-6 is the GRPO/GSPO standard.")
    ap.add_argument("--kl-beta", type=float, default=1e-3,
                    help="KL coefficient against the frozen reference policy.")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--save-steps", type=int, default=50)

    # LoRA
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)

    # Misc
    ap.add_argument("--use-liger", action="store_true", default=True,
                    help="Liger fused-CE kernel (default ON). Saves ~80%% "
                         "LM-head memory, required at 16K and recommended at 8K.")
    ap.add_argument("--no-liger", dest="use_liger", action="store_false")
    ap.add_argument("--report-to", default="wandb",
                    help="Trainer report backend. Set 'none' to disable W&B.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--extra-args", default="",
                    help="Verbatim extra flags passed to `swift rlhf` "
                         "(advanced; rarely needed).")
    args = ap.parse_args()

    phase = PHASE_DEFAULTS[args.phase]
    num_gen = args.num_generations or phase["num_gen"]
    max_completion = phase["max_completion"]
    max_length = 1024 + max_completion  # 1K prompt budget
    steps = args.steps or phase["default_steps"]

    # ms-swift defaults to ModelScope (CN hub). Force HuggingFace lookups so
    # arbitrary HF model ids resolve correctly.
    os.environ.setdefault("USE_HF", "1")
    # Reduce PyTorch CUDA allocator fragmentation. Recommended when an OOM has
    # many "reserved but unallocated" MiB; helps 8K/16K fit alongside vLLM colocate.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Build the loss-specific flags.
    if args.loss == "gspo":
        loss_flags = ["--loss_type", "grpo",
                      "--importance_sampling_level", "sequence"]
    else:  # cispo
        loss_flags = ["--loss_type", "cispo",
                      "--epsilon_high", str(args.eps_high)]

    cmd = [
        "swift", "rlhf",
        "--rlhf_type", "grpo",
        *loss_flags,
        "--beta", str(args.kl_beta),
        "--model", args.model,
        "--model_type", "qwen3",
        "--template", "qwen3_thinking",
        "--lora_rank", str(args.lora_rank),
        "--lora_alpha", str(args.lora_alpha),
        "--target_modules", "all-linear",
        "--torch_dtype", "bfloat16",
        "--attn_impl", "flash_attn",
        "--gradient_checkpointing", "true",
        "--dataset", args.dataset,
        "--reward_funcs", "accuracy",
        "--num_generations", str(num_gen),
        "--max_length", str(max_length),
        "--max_completion_length", str(max_completion),
        "--temperature", "0.7",
        "--top_p", "0.95",
        "--top_k", "20",
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", str(args.grad_accum),
        "--learning_rate", str(args.lr),
        "--lr_scheduler_type", "cosine",
        "--warmup_ratio", "0.03",
        "--max_steps", str(steps),
        "--save_steps", str(args.save_steps),
        "--save_total_limit", "-1",
        "--logging_steps", "1",
        "--use_vllm", "true",
        "--vllm_mode", "colocate",
        "--vllm_gpu_memory_utilization", "0.4",
        "--vllm_max_model_len", str(max_length),
        "--output_dir", args.output,
        "--report_to", args.report_to,
        "--seed", str(args.seed),
    ]
    if args.use_liger:
        cmd += ["--use_liger_kernel", "true"]
    if args.extra_args.strip():
        cmd += args.extra_args.split()

    print(">>> RL training")
    print(f"    loss       = {args.loss}")
    print(f"    phase      = {args.phase} (max_completion={max_completion}, num_gen={num_gen})")
    print(f"    steps      = {steps}")
    print(f"    lr / KL    = {args.lr} / {args.kl_beta}")
    print(f"    model      = {args.model}")
    print(f"    dataset    = {args.dataset}")
    print(f"    output     = {args.output}")
    print(f"    use_liger  = {args.use_liger}")
    print()

    sys.stdout.flush()
    proc = subprocess.run(cmd)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
