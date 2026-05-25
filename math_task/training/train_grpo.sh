#!/bin/bash
# CISPO LoRA training for Qwen3-1.7B on math reasoning.
#
# Thin wrapper around ms-swift's `swift rlhf` CLI. Uses CISPO
# (Clipped Importance Sampling Policy Optimization, MiniMax-M1 paper)
# to preserve the gradient signal on low-probability self-correction
# tokens (Wait, Hmm, Aha) that GRPO/PPO clipping would suppress.
#
# Phase schedule (matches the project plan's 4K -> 8K -> 16K curriculum):
#   PHASE=4k   max_completion=4096   num_gen=8   from base
#   PHASE=8k   max_completion=8192   num_gen=8   from 4K checkpoint at plateau
#   PHASE=16k  max_completion=16384  num_gen=4   from 8K checkpoint at plateau
#
# Usage (from repo root):
#   PHASE=4k bash math_task/training/train_grpo.sh \
#       --dataset /scratch/<user>/rl_data/deepscaler_filtered.jsonl \
#       --output  /scratch/<user>/ckpt/math_rl_4k_v1
#
#   # Phase 2, resuming from phase-1 LoRA adapter:
#   PHASE=8k INIT_FROM=/scratch/<user>/ckpt/math_rl_4k_v1/checkpoint-500 \
#       bash math_task/training/train_grpo.sh \
#       --dataset /scratch/<user>/rl_data/deepscaler_filtered.jsonl \
#       --output  /scratch/<user>/ckpt/math_rl_8k_v1
#
# Env overrides:
#   RL_MODEL     : --model path. Default Qwen/Qwen3-1.7B (Track B from-base).
#                  For Track A, set to cs-552-2026-mystery-machine/math_model
#                  (already-merged SFT-200 weights with new system-prompt template).
#   PHASE        : 4k | 8k | 16k    (default 4k)
#   INIT_FROM    : path to previous-phase LoRA checkpoint (default: none = from base)
#   MAX_STEPS    : training steps  (default: pilot=100, full=500/500/300 by phase)
#   PILOT        : 1 to use pilot defaults (100 steps), else full
#   LR           : learning rate (default 1e-6 — CISPO/GRPO is sensitive, lower than SFT)
#   KL_BETA      : KL coeff vs reference model (default 0.001, DeepScaleR's value)
#   EPS_HIGH     : CISPO ε_high (default 5.0 from MiniMax docs)
#   GRAD_ACCUM   : gradient accumulation steps (default 8)
#   EXTRA_ARGS   : passed verbatim to `swift rlhf`

set -euo pipefail

# ms-swift defaults to ModelScope (CN hub). Force HuggingFace lookups so our
# cs-552-2026-mystery-machine/math_model repo resolves correctly.
export USE_HF="${USE_HF:-1}"

PHASE="${PHASE:-4k}"
PILOT="${PILOT:-0}"

case "$PHASE" in
  4k)
    MAX_COMPLETION=4096
    NUM_GEN=8
    DEFAULT_STEPS_FULL=500
    ;;
  8k)
    MAX_COMPLETION=8192
    NUM_GEN=8
    DEFAULT_STEPS_FULL=500
    ;;
  16k)
    MAX_COMPLETION=16384
    NUM_GEN=4
    DEFAULT_STEPS_FULL=300
    ;;
  *)
    echo "PHASE must be one of: 4k, 8k, 16k (got: $PHASE)" >&2
    exit 2
    ;;
esac

if [ "$PILOT" = "1" ]; then
  DEFAULT_STEPS=100
else
  DEFAULT_STEPS="$DEFAULT_STEPS_FULL"
fi

MAX_STEPS="${MAX_STEPS:-$DEFAULT_STEPS}"
LR="${LR:-1e-6}"
KL_BETA="${KL_BETA:-0.001}"
EPS_HIGH="${EPS_HIGH:-5.0}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
INIT_FROM="${INIT_FROM:-}"
RL_MODEL="${RL_MODEL:-Qwen/Qwen3-1.7B}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Parse --dataset and --output from CLI (the only two we make first-class).
DATASET=""
OUTPUT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --output)  OUTPUT="$2";  shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [ -z "$DATASET" ] || [ -z "$OUTPUT" ]; then
  echo "usage: $0 --dataset PATH --output PATH" >&2
  exit 2
fi

mkdir -p "$OUTPUT"

# `max_length` = prompt budget (1024) + completion budget.
MAX_LENGTH=$((1024 + MAX_COMPLETION))

echo ">>> CISPO LoRA training"
echo "    model            = $RL_MODEL"
echo "    phase            = $PHASE (pilot=$PILOT)"
echo "    max_completion   = $MAX_COMPLETION"
echo "    max_length       = $MAX_LENGTH"
echo "    num_generations  = $NUM_GEN"
echo "    max_steps        = $MAX_STEPS"
echo "    lr               = $LR"
echo "    kl_beta          = $KL_BETA"
echo "    cispo eps_high   = $EPS_HIGH"
echo "    grad_accum       = $GRAD_ACCUM"
echo "    dataset          = $DATASET"
echo "    output           = $OUTPUT"
echo "    init_from        = ${INIT_FROM:-(none)}"

ADAPTER_ARG=()
if [ -n "$INIT_FROM" ]; then
  # Continue from a previous LoRA adapter. ms-swift accepts this via --adapters.
  ADAPTER_ARG=(--adapters "$INIT_FROM")
fi

# `--reward_funcs accuracy format` enables ms-swift's built-in math-correctness
# reward (Math-Verify on \boxed{}) plus a small format reward for think/boxed
# structure. Weight format low so accuracy dominates.
exec swift rlhf \
  --rlhf_type grpo \
  --loss_type cispo \
  --epsilon_high "$EPS_HIGH" \
  --beta "$KL_BETA" \
  --model "$RL_MODEL" \
  --model_type qwen3 \
  --template qwen3_thinking \
  --lora_rank 32 \
  --lora_alpha 64 \
  --target_modules all-linear \
  --torch_dtype bfloat16 \
  --attn_impl flash_attn \
  --gradient_checkpointing true \
  --dataset "$DATASET" \
  --reward_funcs accuracy format \
  --reward_weights 1.0 0.1 \
  --num_generations "$NUM_GEN" \
  --max_length "$MAX_LENGTH" \
  --max_completion_length "$MAX_COMPLETION" \
  --temperature 0.7 \
  --top_p 0.95 \
  --top_k 20 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --learning_rate "$LR" \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --max_steps "$MAX_STEPS" \
  --save_steps 50 \
  --save_total_limit -1 \
  --logging_steps 1 \
  --use_vllm true \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization 0.4 \
  --vllm_max_model_len "$MAX_LENGTH" \
  --output_dir "$OUTPUT" \
  --report_to "${REPORT_TO:-wandb}" \
  --seed 42 \
  "${ADAPTER_ARG[@]}" \
  $EXTRA_ARGS
