#!/usr/bin/env bash
set -euo pipefail

HF_TOKEN="${HF_TOKEN:?HF_TOKEN environment variable is required}"
TEACHER_MODEL="models/teacher"
STUDENT_MODEL="models/qwen3_thinking"

log_stats() {
    local file="$1"
    local label="$2"
    python3 - "$file" "$label" <<'PYEOF'
import json, sys
from pathlib import Path

p = Path(sys.argv[1])
label = sys.argv[2]
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
correct = sum(1 for r in rows if r.get("correct") is True)
wrong   = sum(1 for r in rows if r.get("correct") is False)
total   = len(rows)
pct     = f"{correct/total:.1%}" if total else "n/a"
print(f"  {label:<30} total={total:>5}  correct={correct:>5}  wrong={wrong:>5}  acc={pct}")
PYEOF
}

TOTAL_STATS_FILE=$(mktemp)
LOG_FILE="data/preprocessing/distill_all_stats.log"
trap 'rm -f "$TOTAL_STATS_FILE"' EXIT

for train_file in data/preprocessing/MATH/train.jsonl; do
    dir=$(dirname "$train_file")
    # skip benchmarks
    case "$dir" in
        *benchmarks*) continue ;;
        *mmlu_aux) continue ;;
    esac

    echo "=== Processing $dir ==="

    #python training/generate_answers.py "$train_file" \
    #    --model-dir "$STUDENT_MODEL" \
    #    --output "$dir/train_student.jsonl" \
    #    --hf-token "$HF_TOKEN" \
    #    --max-new-tokens 8192 \
    #    --student

    #python training/generate_answers.py "$train_file" \
    #    --model-dir "$TEACHER_MODEL" \
    #    --output "$dir/train_teacher.jsonl" \
    #    --hf-token "$HF_TOKEN" \
    #    --max-new-tokens 8192

    python training/resample_wrong_answers.py "$dir/train_teacher.jsonl" \
        --model-dir "$TEACHER_MODEL" \
        --output "$dir/teacher_corrected.jsonl" \
        --hf-token "$HF_TOKEN"

    python training/resample_wrong_answers.py "$dir/train_student.jsonl" \
        --model-dir "$STUDENT_MODEL" \
        --output "$dir/student_corrected.jsonl" \
        --hf-token "$HF_TOKEN" \
        --student

    python training/merge_reasoning_from_student_teacher.py \
        "$dir/student_corrected.jsonl" \
        "$dir/teacher_corrected.jsonl" \
        --output "$dir/merged_student_teacher.jsonl"

    {
        echo "  --- stats: $dir ---"
        log_stats "$dir/train_student.jsonl"     "student (initial)"
        log_stats "$dir/student_corrected.jsonl" "student (resampled)"
        log_stats "$dir/train_teacher.jsonl"     "teacher (initial)"
        log_stats "$dir/teacher_corrected.jsonl" "teacher (resampled)"
    } | tee -a "$LOG_FILE"
    echo "$dir/train_student.jsonl $dir/student_corrected.jsonl $dir/train_teacher.jsonl $dir/teacher_corrected.jsonl" >> "$TOTAL_STATS_FILE"
done

{ echo ""; echo "=== GLOBAL TOTALS ==="; python3 - "$TOTAL_STATS_FILE" <<'PYEOF'
import json, sys
from pathlib import Path

def load(path):
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

def stats(rows, label):
    correct = sum(1 for r in rows if r.get("correct") is True)
    wrong   = sum(1 for r in rows if r.get("correct") is False)
    total   = len(rows)
    pct     = f"{correct/total:.1%}" if total else "n/a"
    print(f"  {label:<30} total={total:>6}  correct={correct:>6}  wrong={wrong:>6}  acc={pct}")

buckets = {"student (initial)": [], "student (resampled)": [], "teacher (initial)": [], "teacher (resampled)": []}
labels = list(buckets.keys())
for line in Path(sys.argv[1]).read_text().splitlines():
    line = line.strip()
    if not line:
        continue
    paths = line.split()
    for path, label in zip(paths, labels):
        buckets[label].extend(load(path))

for label, rows in buckets.items():
    stats(rows, label)
PYEOF
} | tee -a "$LOG_FILE"

python training/lora_post_train.py \
    --dataset-file data/preprocessing/teacher_distillation.jsonl \
    --model-folder models/qwen3_thinking \
    --output-dir models/full_sft_math \
    --run-name sft_full_math \
    --wandb-key "${WANDB_API_KEY:?WANDB_API_KEY environment variable is required}" \
    --merge-and-save \
    --overwrite-output

python training/lora_post_train.py \
    --model-folder models/qwen3_thinking \
    --adapter-folder models/full_sft_math/lora_adapter/checkpoint-4198 \
    --dataset-file data/preprocessing/swissai_merged_sft.jsonl \
    --output-dir models/full_sft_math \
    --wandb-key "${WANDB_API_KEY:?WANDB_API_KEY environment variable is required}" \
    --merge-and-save \
    --append

python training/grpo_train.py \
  --model-folder models/qwen3_thinking \
  --dataset-file data/preprocessing/student_wrong.jsonl \
  --output-dir models/full_math_grpo \
  --adapter-folder models/full_sft_math/lora_adapter/checkpoint-93 \
  --overwrite-output \
  --save-strategy steps \
  --save-steps 10000

python training/grpo_train.py \
  --model-folder models/full_sft \
  --dataset-file data/preprocessing/grpo_runs/partial_success.jsonl \
  --output-dir models/full_math_grpo \
  --use-lora \
  --overwrite-output \
  --save-strategy steps \
  --save-steps 10000 \
  --learning-rate 5e-6 \
  --beta 0.02 \
  --format-weight 0.1 \
  --acc-weight 0.9 \
  --num-generations 16

python training/generate_answers.py data/preprocessing/full_train.jsonl \
    --model-dir models/full_sft \
    --output data/preprocessing/grpo_runs/run1.jsonl \
    --hf-token "$HF_TOKEN" \
    --max-new-tokens 8192 \
    --seed 42 \
    --student \
    --max-model-len 10240 \
    --max-num-seqs 32

python training/generate_answers.py data/preprocessing/full_train.jsonl \
    --model-dir models/full_sft \
    --output data/preprocessing/grpo_runs/run2.jsonl \
    --hf-token "$HF_TOKEN" \
    --max-new-tokens 8192 \
    --seed 123 \
    --student \
    --max-model-len 10240 \
    --max-num-seqs 32

python training/generate_answers.py data/preprocessing/full_train.jsonl \
    --model-dir models/full_sft \
    --output data/preprocessing/grpo_runs/run3.jsonl \
    --hf-token "$HF_TOKEN" \
    --max-new-tokens 8192 \
    --seed 999 \
    --student \
    --max-model-len 10240 \
    --max-num-seqs 64

python training/dpo_train.py \
    --model-folder models/full_sft \
    --dataset-file data/preprocessing/dpo_pairs.jsonl \
    --output-dir models/full_sft_dpo \
    --beta 0.1 \
    --learning-rate 3e-6 \
    --num-train-epochs 1.0 \
    --per-device-train-batch-size 1 \
    --gradient-accumulation-steps 8 \
    --overwrite-output

