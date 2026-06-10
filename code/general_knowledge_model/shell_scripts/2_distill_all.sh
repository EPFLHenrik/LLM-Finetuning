#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GK_DIR="$(dirname "$SCRIPT_DIR")"               # general_knowledge_model/
PROJECT_DIR="$(dirname "$(dirname "$GK_DIR")")"  # project root
DATA_DIR="$PROJECT_DIR/data/general_knowledge"
TRAINING_DIR="$GK_DIR/training"
PREPROCESSING_DIR="$DATA_DIR/processed"

HF_TOKEN="${HF_TOKEN:?HF_TOKEN environment variable is required}"
TEACHER_MODEL="$PROJECT_DIR/checkpoints/general_knowledge_model/teacher"
STUDENT_MODEL="$PROJECT_DIR/checkpoints/general_knowledge_model/qwen_thinking3"

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
LOG_FILE="$PREPROCESSING_DIR/distill_all_stats.log"
trap 'rm -f "$TOTAL_STATS_FILE"' EXIT

for train_file in "$PREPROCESSING_DIR/train_datasets"/*/train.jsonl; do
    dir=$(dirname "$train_file")
    # skip benchmarks
    case "$dir" in
        *benchmarks*) continue ;;
        *mmlu_aux) continue ;;
    esac

    echo "=== Processing $dir ==="

    python "$TRAINING_DIR/generate_answers.py" "$train_file" \
        --model-dir "$STUDENT_MODEL" \
        --output "$dir/train_student.jsonl" \
        --hf-token "$HF_TOKEN" \
        --max-new-tokens 8192 \
        --student

    python "$TRAINING_DIR/generate_answers.py" "$train_file" \
        --model-dir "$TEACHER_MODEL" \
        --output "$dir/train_teacher.jsonl" \
        --hf-token "$HF_TOKEN" \
        --max-new-tokens 8192

    python "$TRAINING_DIR/resample_wrong_answers.py" "$dir/train_teacher.jsonl" \
        --model-dir "$TEACHER_MODEL" \
        --output "$dir/teacher_corrected.jsonl" \
        --hf-token "$HF_TOKEN"

    python "$TRAINING_DIR/resample_wrong_answers.py" "$dir/train_student.jsonl" \
        --model-dir "$STUDENT_MODEL" \
        --output "$dir/student_corrected.jsonl" \
        --hf-token "$HF_TOKEN" \
        --student

    python "$TRAINING_DIR/merge_reasoning_from_student_teacher.py" \
        "$dir/student_corrected.jsonl" \
        "$dir/teacher_corrected.jsonl" \
        --output "$dir/merged_student_teacher.jsonl" \
        --distillation-output "$PREPROCESSING_DIR/merged_distillations.jsonl" \
        --wrong-output "$PREPROCESSING_DIR/wrong_distillations.jsonl"

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
