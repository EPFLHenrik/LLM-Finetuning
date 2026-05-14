#!/usr/bin/env bash

# Run from the repository root.

# 1. Export debug checkpoint locally
python -m multilingual.train.export \
  --adapter_path multilingual/results/checkpoints/debug_sft_newdata/best_adapter \
  --output_dir multilingual/results/exports/debug_export


# 2. Export debug checkpoint and push to HF
python -m multilingual.train.export \
  --adapter_path multilingual/results/checkpoints/debug_sft_newdata/best_adapter \
  --output_dir multilingual/results/exports/debug_export \
  --hf_repo cs-552-2026-mystery-machine/multilingual_model \
  --push


# 3. Export real checkpoint locally
python -m multilingual.train.export \
  --adapter_path multilingual/results/checkpoints/sft_boxed_lora_baseline/best_adapter \
  --output_dir multilingual/results/exports/sft_boxed_lora_baseline


# 4. Export real checkpoint and push to HF
python -m multilingual.train.export \
  --adapter_path multilingual/results/checkpoints/sft_boxed_lora_baseline/best_adapter \
  --output_dir multilingual/results/exports/sft_boxed_lora_baseline \
  --hf_repo cs-552-2026-mystery-machine/multilingual_model \
  --push