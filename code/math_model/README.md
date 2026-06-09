## Math Model Reproducibility Guide

This document describes the exact pipeline used to reproduce the submitted math model.

The commands below are intentionally kept in the original order and should be executed sequentially.

## Goal

Reproduce the final submitted math model by:
- preparing the raw RL problem pool (DeepScaleR),
- building the SFT corpus (multi-source CoT + distilled AIME/AMC traces),
- training an SFT adapter (LoRA on the curated CoT mixture),
- learnability-filtering the RL pool against the base model,
- training the first GSPO phase on top of the SFT checkpoint,
- merging the phase-1 adapter and re-filtering the pool against the phase-1 policy,
- training the second GSPO phase on the re-filtered pool,
- merging the phase-2 adapter and pushing it to Hugging Face,
- benchmarking the final checkpoint.

## Before You Start

- Run all commands from the repository root.
- Ensure your Python environment is activated and project dependencies are installed (see [requirements.txt](../../requirements.txt)).
- For ms-swift (used by `rl.py`): `pip install "ms-swift>=3.0" "math-verify" "liger-kernel>=0.4" "bitsandbytes>=0.45"`.
- All training stages assume a single NVIDIA A100 (40 GB). The 8K RL phases require Liger Kernel + `num_generations=4` to fit in 40 GB.
- Substitute `<user>` with your username (or any working directory on shared `/scratch`).

## 1. Build the SFT corpus

Two-phase pipeline: pull all upstream chain-of-thought datasets, generate distilled traces for the AIME/AMC tail, then finalise the 30K mixed corpus.

### 1.1 Pull CoT sources + emit the "needs distillation" list

```
python -m code.math_model.preprocessing.sft.build_corpus \
    --phase prep \
    --workdir data/math/sft/workdir
```

This downloads OpenR1-Math-220k, NuminaMath-CoT, GAIR/LIMO, simplescaling/s1K-1.1 and nvidia/OpenMathReasoning, filters each to rows whose assistant trace has `<think>...</think>` + extractable `\boxed{}` + length in `[1500, 8000]` Qwen3 tokens, and writes one `*_cot.jsonl` per source. It also emits `needs_distill.jsonl` containing AIME 2024 / AIME 2025 / AMC 2023 problems that lack CoT traces.

### 1.2 Distil reasoning traces with a stronger teacher

```
python -m code.math_model.preprocessing.sft.distill_traces \
    --input  data/math/sft/workdir/needs_distill.jsonl \
    --output data/math/sft/workdir/distilled.jsonl \
    --teacher Valdemardi/DeepSeek-R1-Distill-Qwen-32B-AWQ \
    --n-samples 8 --keep-top-k 4 --max-tokens 8192
```

vLLM samples eight candidate traces per problem from a 32 B AWQ-quantised teacher, rejection-samples on Math-Verify against the gold answer, and keeps up to four correct traces per problem. The job is resumable (any partial output is detected and skipped).

### 1.3 Finalise the 30K mixed corpus

```
python -m code.math_model.preprocessing.sft.build_corpus \
    --phase finalize \
    --workdir data/math/sft/workdir \
    --output  data/math/sft/sft_full.jsonl
```

This stratified-samples each source to the target count, deduplicates by normalised problem fingerprint, and writes the final SFT JSONL with `{messages: [system, user, assistant], source}`.

Expected artifact:
- `data/math/sft/sft_full.jsonl` (~30K rows).
- `data/math/sft/sft_full_stats.json` (per-source counts and length stats).

## 2. Prepare the raw RL problem pool

```
python -m code.math_model.preprocessing.rl.prepare_pool \
    --output data/math/rl/deepscaler_full.jsonl
```

Pulls `agentica-org/DeepScaleR-Preview-Dataset` (~40K problems, deduplicated upstream against MATH-500 and AIME 2024/2025) and normalises every row to `{messages: [user], solution, source}`.

Expected artifact:
- `data/math/rl/deepscaler_full.jsonl` (~40K rows).

## 3. SFT

Train the LoRA adapter on the curated CoT mixture.

```
python -m code.math_model.sft \
    --data    data/math/sft/sft_full.jsonl \
    --output  checkpoints/math_model/sft \
    --epochs 2 \
    --lr 2e-5 \
    --grad-accum 8 \
    --save-steps 200 \
    --max-length 8192
```

This trains LoRA `r=32, α=64` on all linear projections in bf16 with Flash-Attention 2 + gradient checkpointing. Every 200 steps the trainer saves an adapter; we use `checkpoint-600` (the lowest-step plateau, see report Table on SFT plateau analysis) as the RL starting point.

Expected artifacts:
- `checkpoints/math_model/sft/checkpoint-200`, `checkpoint-400`, ..., `checkpoint-3448`
- `checkpoints/math_model/sft/training_args.json`

### 3.1 Merge the chosen SFT checkpoint into the base weights

```
python -m code.math_model.huggingface.push_checkpoint \
    checkpoints/math_model/sft/checkpoint-600 \
    --local-out checkpoints/math_model/sft_merged \
    --skip-vllm-check
```

(Append `--repo cs-552-2026-mystery-machine/math_model_sft` if you also want to push the SFT-only baseline to a separate HF repo.)

Expected artifact:
- `checkpoints/math_model/sft_merged/` (full Qwen3-1.7B + merged SFT LoRA, ~3.4 GB `model.safetensors`).

## 4. Build the phase-1 RL training pool

Filter the raw DeepScaleR pool against base Qwen3-1.7B using the offline learnability filter, then prepend the team system prompt so the training distribution matches the inference-time chat template.

### 4.1 Learnability filter (against base Qwen3-1.7B)

```
python -m code.math_model.preprocessing.rl.filter_learnability \
    --input  data/math/rl/deepscaler_full.jsonl \
    --output data/math/rl/phase1_filtered.jsonl \
    --model  Qwen/Qwen3-1.7B \
    --n-samples 8 --k-lo 1 --k-hi 6 --max-tokens 4096
```

vLLM runs 8 rollouts per problem at 4K context (T=0.7, top-p 0.95, top-k 20), scores each completion with Math-Verify on the extracted `\boxed{}` answer, and keeps problems with a base pass-count in `[1, 6]/8`. Expect ~1,219 retained rows.

### 4.2 Prepend the system prompt

```
python -m code.math_model.preprocessing.rl.add_system_prompt \
    --input  data/math/rl/phase1_filtered.jsonl \
    --output data/math/rl/phase1_filtered_sys.jsonl
```

Expected artifact:
- `data/math/rl/phase1_filtered_sys.jsonl` (~1,219 rows with `messages = [system, user]`).

## 5. RL training — phase 1 (GSPO from the SFT checkpoint)

```
python -m code.math_model.rl \
    --model    checkpoints/math_model/sft_merged \
    --dataset  data/math/rl/phase1_filtered_sys.jsonl \
    --output   checkpoints/math_model/rl_phase1 \
    --loss gspo \
    --phase 8k \
    --num-generations 4 \
    --steps 500
```

The `gspo` loss is the shipped default (sequence-level importance sampling via the TRL flag `--importance_sampling_level sequence`). All other hyperparameters match the report Table on math training hyperparameters (LR `1e-6`, KL β `1e-3`, save every 50 steps, Liger Kernel ON).

Expected artifacts:
- `checkpoints/math_model/rl_phase1/checkpoint-{50,100,...,500}`

### 5.1 Merge the phase-1 LoRA into the base

```
python -m code.math_model.huggingface.push_checkpoint \
    checkpoints/math_model/rl_phase1/checkpoint-500 \
    --base-model checkpoints/math_model/sft_merged \
    --local-out  checkpoints/math_model/rl_phase1_merged \
    --skip-vllm-check
```

The `--base-model` flag is critical: the phase-1 LoRA must be merged into the SFT-merged weights, not bare Qwen3-1.7B (which would discard everything SFT learned).

Expected artifact:
- `checkpoints/math_model/rl_phase1_merged/` (~3.4 GB).

## 6. Re-build the RL training pool against the phase-1 policy

The original `[1, 6]/8` band was computed against base Qwen3-1.7B. After ~500 GSPO steps the policy has drifted; many "learnable for base" problems are now solved on every rollout (or never) and contribute zero gradient. Re-running the filter against the trained policy restores variance.

### 6.1 Learnability filter (against the phase-1 checkpoint)

```
python -m code.math_model.preprocessing.rl.filter_learnability \
    --input  data/math/rl/deepscaler_full.jsonl \
    --output data/math/rl/phase2_filtered.jsonl \
    --model  checkpoints/math_model/rl_phase1_merged \
    --n-samples 8 --k-lo 1 --k-hi 6 --max-tokens 4096
```

Expect ~1,235 retained rows (comparable size, different problem set).

### 6.2 Prepend the system prompt

```
python -m code.math_model.preprocessing.rl.add_system_prompt \
    --input  data/math/rl/phase2_filtered.jsonl \
    --output data/math/rl/phase2_filtered_sys.jsonl
```

## 7. RL training — phase 2 (GSPO from the phase-1 checkpoint)

```
python -m code.math_model.rl \
    --model    checkpoints/math_model/rl_phase1_merged \
    --dataset  data/math/rl/phase2_filtered_sys.jsonl \
    --output   checkpoints/math_model/rl_phase2 \
    --loss gspo \
    --phase 8k \
    --num-generations 4 \
    --steps 800
```

Expected artifacts:
- `checkpoints/math_model/rl_phase2/checkpoint-{50,100,...,800}`

### 7.1 Merge the phase-2 LoRA and push to Hugging Face

```
python -m code.math_model.huggingface.push_checkpoint \
    checkpoints/math_model/rl_phase2/checkpoint-800 \
    --base-model checkpoints/math_model/rl_phase1_merged \
    --local-out  checkpoints/math_model/final \
    --repo cs-552-2026-mystery-machine/math_model
```

This script merges the LoRA into the phase-1-merged weights, attaches the team chat template + generation config, smoke-tests the result with vLLM, and pushes the full model to `cs-552-2026-mystery-machine/math_model` so the CI grader can load it.

Expected artifact:
- HF model at `cs-552-2026-mystery-machine/math_model` (the final shipped checkpoint).
- `checkpoints/math_model/final/` (local mirror).

## 8. Benchmarking

To evaluate any checkpoint on the calibrated math benchmark, run:

```
python -m code.math_model.benchmark \
    --name <my-model-name> \
    --checkpoint <checkpoint_dir_or_HF_id> \
    --benchmark <broad|full|quick>
```

Replace `--benchmark <...>` with `--run_all` to run both `broad` and `full` profiles back-to-back. The defaults (`--n 8 --budget final`) match the classroom CI grading contract (pass@8 at 16K tokens).

Examples:

```
# Final shipped checkpoint, broad profile (~3.5 h on a single A100):
python -m code.math_model.benchmark \
    --name shipped \
    --checkpoint cs-552-2026-mystery-machine/math_model \
    --benchmark broad

# Base baseline for the report's main comparison:
python -m code.math_model.benchmark \
    --name base \
    --checkpoint Qwen/Qwen3-1.7B \
    --benchmark broad
```

Results are written to `code/math_model/results/eval_v2/<model>_<budget>_<profile>_think-on_n-<n>.json` and a one-line summary is appended to `code/math_model/RUNS.md`.

The benchmark subsets and weights are documented in the report (`MATH-500` stratified + `MATH-500-hard` + `AIME 2024/2025` + `GSM8K` for `broad`, plus `OlympiadBench-Math` + `Hendrycks-MATH-L4/5` for `full`); the materialised JSONL versions are also released as part of the team's public dataset at `cs-552-2026-mystery-machine/math` under `benchmarks/`.
