# Math Reasoning Model

A math-reasoning model built on `Qwen/Qwen3-1.7B`, trained with a two-stage
recipe and evaluated with weighted pass@k across standard competition-math
benchmarks.

## Overview

The model is fine-tuned from Qwen3-1.7B in two stages:

1. **Supervised fine-tuning (SFT)** — a LoRA adapter trained on a ~30K curated
   chain-of-thought mixture (multi-source CoT + teacher-distilled AIME/AMC
   traces).
2. **Reinforcement learning with verifiable rewards (RLVR)** — two GSPO phases
   on a learnability-filtered DeepScaleR pool, each re-filtered against the
   current policy so the training distribution keeps non-zero reward variance.

At inference the model produces a `<think>...</think>` reasoning trace followed
by a single `\boxed{}` final answer (enforced by the system prompt in
[constants/system_prompt.py](constants/system_prompt.py), mirrored in
[huggingface/chat_template.jinja](huggingface/chat_template.jinja)). Evaluation
computes weighted pass@1 and pass@8 at a 16K-token budget via Math-Verify on the
`\boxed{}` answer.

## Repository Structure

```
math_model/
├── sft.py                          # LoRA SFT entry point
├── rl.py                           # GRPO/GSPO RL entry point (wraps ms-swift)
├── constants/
│   └── system_prompt.py            # shared system prompt (single source of truth)
├── preprocessing/
│   ├── sft/
│   │   ├── build_corpus.py         # build the multi-source SFT CoT corpus
│   │   └── distill_traces.py       # teacher-distill traces for AIME/AMC
│   └── rl/
│       ├── prepare_pool.py         # DeepScaleR -> RL problem pool
│       ├── filter_learnability.py  # keep [k_lo, k_hi]/N learnable problems
│       └── add_system_prompt.py    # prepend the system message to RL rows
├── evaluation/
│   ├── evaluate.py                 # weighted multi-subset evaluator
│   ├── run_benchmark.py            # profile-level benchmark CLI
│   ├── config.py                   # token budgets + EvalConfig
│   ├── dataset_loader.py           # benchmark loaders
│   ├── generation.py               # vLLM batch generation
│   ├── scoring.py                  # pass@k scoring
│   ├── pass_at_k.py                # unbiased pass@k estimator
│   ├── diagnostics.py              # box-compliance / truncation / length metrics
│   ├── answer_extraction.py        # \boxed{} extraction + math equivalence
│   ├── answer_matching.py          # benchmark answer extraction/comparison
│   └── materialize_benchmarks.py   # export a profile to the HuggingFace Hub (benchmarks/*.jsonl)
└── huggingface/
    ├── push_checkpoint.py          # merge a LoRA, save locally, optionally push to HF
    └── chat_template.jinja         # chat template (embeds the system prompt)
```

## Before You Start

- Run all commands from the repository root.
- Activate your Python environment and install project dependencies (see
  [requirements.txt](../../requirements.txt)).
- For ms-swift (used by `rl.py`):
  `pip install "ms-swift>=3.0" "math-verify" "liger-kernel>=0.4" "bitsandbytes>=0.45"`.
- All training stages assume a single NVIDIA A100 (40 GB). The 8K RL phases
  require Liger Kernel + `num_generations=4` to fit in 40 GB.

The full pipeline below is ordered and should be run sequentially.

## Dataset Preparation

### 1. Build the SFT corpus

Two-phase pipeline: pull all upstream chain-of-thought datasets, distill traces
for the AIME/AMC tail, then finalize the ~30K mixed corpus.

#### 1.1 Pull CoT sources + emit the "needs distillation" list

```
python -m code.math_model.preprocessing.sft.build_corpus \
    --phase prep \
    --workdir data/math/sft/workdir
```

Downloads OpenR1-Math-220k, NuminaMath-CoT, GAIR/LIMO, simplescaling/s1K-1.1 and
nvidia/OpenMathReasoning, filters each to rows whose assistant trace has
`<think>...</think>` + extractable `\boxed{}` + length in `[1500, 8000]` Qwen3
tokens, and writes one `*_cot.jsonl` per source. It also emits
`needs_distill.jsonl` containing AIME 2024 / AIME 2025 / AMC 2023 problems that
lack CoT traces.

#### 1.2 Distil reasoning traces with a stronger teacher

```
python -m code.math_model.preprocessing.sft.distill_traces \
    --input  data/math/sft/workdir/needs_distill.jsonl \
    --output data/math/sft/workdir/distilled.jsonl \
    --teacher Valdemardi/DeepSeek-R1-Distill-Qwen-32B-AWQ \
    --n-samples 8 --keep-top-k 4 --max-tokens 8192
```

vLLM samples eight candidate traces per problem from a 32B AWQ-quantized teacher,
rejection-samples on Math-Verify against the gold answer, and keeps up to four
correct traces per problem. The job is resumable (partial output is detected and
skipped).

#### 1.3 Finalize the ~30K mixed corpus

```
python -m code.math_model.preprocessing.sft.build_corpus \
    --phase finalize \
    --workdir data/math/sft/workdir \
    --output  data/math/sft/sft_full.jsonl
```

Stratified-samples each source to its target count, deduplicates by normalized
problem fingerprint, and writes the final SFT JSONL with
`{messages: [system, user, assistant], source}`.

Artifacts:
- `data/math/sft/sft_full.jsonl` (~30K rows)
- `data/math/sft/sft_full_stats.json` (per-source counts and length stats)

### 2. Prepare the raw RL problem pool

```
python -m code.math_model.preprocessing.rl.prepare_pool \
    --output data/math/rl/deepscaler_full.jsonl
```

Pulls `agentica-org/DeepScaleR-Preview-Dataset` (~40K problems, deduplicated
upstream against MATH-500 and AIME 2024/2025) and normalizes every row to
`{messages: [user], solution, source}`.

Artifact:
- `data/math/rl/deepscaler_full.jsonl` (~40K rows)

## Training

### 3. SFT

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

Trains LoRA `r=16, α=32` on all linear projections in bf16 with Flash-Attention 2
+ gradient checkpointing. The adapter is saved every 200 steps; `checkpoint-600`
(the lowest-step plateau) is used as the RL starting point.

Artifacts:
- `checkpoints/math_model/sft/checkpoint-200`, `checkpoint-400`, ...
- `checkpoints/math_model/sft/training_args.json`

#### 3.1 Merge the chosen SFT checkpoint into the base weights

```
python -m code.math_model.huggingface.push_checkpoint \
    checkpoints/math_model/sft/checkpoint-600 \
    --local-out checkpoints/math_model/sft_merged \
    --skip-vllm-check
```

Without `--repo` the merged model is saved locally only (no push). Append
`--repo <user>/<repo>` to also publish an SFT-only baseline.

Artifact:
- `checkpoints/math_model/sft_merged/` (full Qwen3-1.7B + merged SFT LoRA, ~3.4 GB)

### 4. Build the phase-1 RL training pool

Filter the raw DeepScaleR pool against base Qwen3-1.7B with the offline
learnability filter, then prepend the system prompt so the training distribution
matches the inference-time chat template.

#### 4.1 Learnability filter (against base Qwen3-1.7B)

```
python -m code.math_model.preprocessing.rl.filter_learnability \
    --input  data/math/rl/deepscaler_full.jsonl \
    --output data/math/rl/phase1_filtered.jsonl \
    --model  Qwen/Qwen3-1.7B \
    --n-samples 8 --k-lo 1 --k-hi 6 --max-tokens 4096
```

vLLM runs 8 rollouts per problem at 4K context (T=0.7, top-p 0.95, top-k 20),
scores each completion with Math-Verify on the extracted `\boxed{}` answer, and
keeps problems with a base pass-count in `[1, 6]/8`. Expect ~1,219 retained rows.

#### 4.2 Prepend the system prompt

```
python -m code.math_model.preprocessing.rl.add_system_prompt \
    --input  data/math/rl/phase1_filtered.jsonl \
    --output data/math/rl/phase1_filtered_sys.jsonl
```

Artifact:
- `data/math/rl/phase1_filtered_sys.jsonl` (~1,219 rows with `messages = [system, user]`)

### 5. RL training — phase 1 (GSPO from the SFT checkpoint)

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

The `gspo` loss is the default (sequence-level importance sampling via TRL's
`--importance_sampling_level sequence`). Other hyperparameters: LR `1e-6`,
KL β `1e-3`, save every 50 steps, Liger Kernel ON.

Artifacts:
- `checkpoints/math_model/rl_phase1/checkpoint-{50,100,...,500}`

#### 5.1 Merge the phase-1 LoRA into the base

```
python -m code.math_model.huggingface.push_checkpoint \
    checkpoints/math_model/rl_phase1/checkpoint-500 \
    --base-model checkpoints/math_model/sft_merged \
    --local-out  checkpoints/math_model/rl_phase1_merged \
    --skip-vllm-check
```

`--base-model` is critical: the phase-1 LoRA must merge into the SFT-merged
weights, not bare Qwen3-1.7B (which would discard everything SFT learned).

Artifact:
- `checkpoints/math_model/rl_phase1_merged/` (~3.4 GB)

### 6. Re-build the RL pool against the phase-1 policy

The original `[1, 6]/8` band was computed against base Qwen3-1.7B. After ~500
GSPO steps the policy has drifted; many "learnable for base" problems are now
solved on every rollout (or never) and contribute zero gradient. Re-running the
filter against the trained policy restores variance.

#### 6.1 Learnability filter (against the phase-1 checkpoint)

```
python -m code.math_model.preprocessing.rl.filter_learnability \
    --input  data/math/rl/deepscaler_full.jsonl \
    --output data/math/rl/phase2_filtered.jsonl \
    --model  checkpoints/math_model/rl_phase1_merged \
    --n-samples 8 --k-lo 1 --k-hi 6 --max-tokens 4096
```

Expect ~1,235 retained rows (comparable size, different problem set).

#### 6.2 Prepend the system prompt

```
python -m code.math_model.preprocessing.rl.add_system_prompt \
    --input  data/math/rl/phase2_filtered.jsonl \
    --output data/math/rl/phase2_filtered_sys.jsonl
```

### 7. RL training — phase 2 (GSPO from the phase-1 checkpoint)

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

Artifacts:
- `checkpoints/math_model/rl_phase2/checkpoint-{50,100,...,800}`

#### 7.1 Merge the phase-2 LoRA and push to Hugging Face

```
python -m code.math_model.huggingface.push_checkpoint \
    checkpoints/math_model/rl_phase2/checkpoint-800 \
    --base-model checkpoints/math_model/rl_phase1_merged \
    --local-out  checkpoints/math_model/final \
    --repo cs-552-2026-mystery-machine/math_model
```

Merges the LoRA into the phase-1-merged weights, attaches the chat template +
generation config, smoke-tests the result with vLLM, and pushes the full model
to `cs-552-2026-mystery-machine/math_model` so it can be served with vLLM.

Artifacts:
- HF model at `cs-552-2026-mystery-machine/math_model` (final checkpoint)
- `checkpoints/math_model/final/` (local mirror)

## Evaluation

To evaluate any checkpoint on the calibrated math benchmark:

```
python -m code.math_model.evaluation.run_benchmark \
    --name <my-model-name> \
    --checkpoint <checkpoint_dir_or_HF_id> \
    --benchmark <broad|full|quick>
```

Use `--run_all` instead of `--benchmark <...>` to run the `broad` then `full`
profiles back-to-back. The defaults (`--n 8 --budget final`) compute pass@8 at a
16K-token budget.

Profiles:
- **broad** (~290 problems): MATH-500 stratified + MATH-500-hard + AIME 2024/2025
  + GSM8K. Used for fast checkpoint ranking.
- **full** (~400 problems): broad + OlympiadBench + Hendrycks-MATH (level 4-5).
  Used for the final model-selection decision.

The lower-level [evaluate.py](evaluation/evaluate.py) exposes finer control
(`--profile`, `--subsets`, `--n-per-subset`, `--temperature`, `--out-dir`) if you
need it.

## Reproducing Results

```
# Final checkpoint, broad profile (~3.5 h on a single A100):
python -m code.math_model.evaluation.run_benchmark \
    --name final \
    --checkpoint cs-552-2026-mystery-machine/math_model \
    --benchmark broad

# Base model baseline for the main comparison:
python -m code.math_model.evaluation.run_benchmark \
    --name base \
    --checkpoint Qwen/Qwen3-1.7B \
    --benchmark broad
```

## Results

Evaluated on the `broad` profile at the 16K budget with `n = 8` (pass@8). The raw
evaluator outputs for every stage below are committed under
[results/eval/](results/eval/) (see its [README](results/eval/README.md) for the
file-to-checkpoint mapping).

**Lineage (weighted over the broad subsets):**

| Stage | weighted pass@1 | weighted pass@8 | MATH-500 (L1–5) pass@8 |
|---|---|---|---|
| Base Qwen3-1.7B | 0.490 | 0.655 | 0.810 |
| + SFT (checkpoint-600) | 0.424 | 0.625 | 0.890 |
| + GSPO phase 1 | 0.417 | 0.645 | 0.940 |
| + GSPO phase 2 (**final**) | 0.419 | **0.684** | 0.910 |

**Per-subset, base vs. final (pass@1 / pass@8):**

| Subset | n | Base | Final |
|---|---|---|---|
| MATH-500 (L1–5) | 100 | 0.757 / 0.810 | 0.771 / 0.910 |
| MATH-500 (L5)   | 80  | 0.583 / 0.688 | 0.531 / 0.800 |
| AIME 2024       | 30  | 0.346 / 0.633 | 0.242 / 0.667 |
| AIME 2025       | 30  | 0.325 / 0.500 | 0.212 / 0.433 |
| GSM8K           | 50  | 0.912 / 0.940 | 0.853 / 0.960 |
| **Weighted**    |     | **0.490 / 0.655** | **0.419 / 0.684** |

The two-stage recipe lifts the headline weighted **pass@8 from 0.655 to 0.684**.
SFT enforces the `\boxed{}` format and raises pass@8 on the easier and
mid-difficulty subsets (MATH-500 L1–5 pass@8 0.810 → 0.890), but trades down
pass@1 on the hardest competition sets — a coverage/confidence trade-off in which
diverse sampling (pass@8) benefits more than greedy-style pass@1. GSPO phase 1
sharpens mid-difficulty pass@8 (MATH-500 L1–5 → 0.940) yet leaves AIME coverage
flat; re-filtering the RL pool against the phase-1 policy and running the second
GSPO phase recovers AIME pass@8 (AIME 2024 0.533 → 0.667) and gives the best
weighted pass@8 (0.684) — the shipped checkpoint. (The format alignment from SFT
is also what drives the large gain on the external 4K-budget pass@1 metric.)

## Outputs

**Checkpoints** (under `checkpoints/math_model/`):
- `sft/checkpoint-*`, `sft_merged/` — SFT adapters and the merged SFT model.
- `rl_phase1/checkpoint-*`, `rl_phase1_merged/` — phase-1 RL.
- `rl_phase2/checkpoint-*`, `final/` — phase-2 RL and the final merged model.

**Metrics** (written by the evaluator):
- `code/math_model/results/eval/<model>_<budget>_<profile>_think-<on|off>_n-<n>.json`
  — full per-subset and per-problem results, including the weighted aggregate.
  The committed lineage artifacts in [results/eval/](results/eval/) follow this
  schema.
- `code/math_model/RUNS.md` — one summary row appended per evaluation run.

**Released artifacts** (Hugging Face):
- Model: `cs-552-2026-mystery-machine/math_model` (LoRA-merged, vLLM-ready).
- Dataset: `cs-552-2026-mystery-machine/math` — raw RL pool, SFT corpus, RL
  phase pools, and the materialized benchmark profiles under `benchmarks/`
  (`broad.jsonl`, `full.jsonl`). Export a profile with
  [materialize_benchmarks.py](evaluation/materialize_benchmarks.py).

## Notes

- **Answer format.** The system prompt requires exactly one `\boxed{}` answer.
  Extraction and math equivalence live in
  [answer_extraction.py](evaluation/answer_extraction.py); pass@k uses the
  unbiased estimator `1 - C(n-c, k) / C(n, k)`.
- **Train/eval parity.** Training and evaluation render prompts through the same
  chat template, so the rollout distribution matches what is served at eval time.
- **Memory.** RL at 8K context fits a 40 GB A100 only with Liger Kernel and
  `num_generations=4`; vLLM runs colocated at `gpu_memory_utilization=0.4`.
- **Determinism.** Loaders, sampling, and training are seeded (default `42`),
  but vLLM/GPU nondeterminism means pass@k can vary by a few tenths of a point
  between runs.
