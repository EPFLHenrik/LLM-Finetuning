#!/usr/bin/env bash
set -euo pipefail

HF_TOKEN="<HF_TOKEN_REDACTED>"
TEACHER_MODEL="models/teacher32"

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

for train_file in data/preprocessing/*/train.jsonl; do
    dir=$(dirname "$train_file")
    case "$dir" in
        *benchmarks*) continue ;;
        *mmlu_aux) continue ;;
    esac

    echo "=== Processing $dir ==="

    python training/generate_answers.py "$train_file" \
        --model-dir "$TEACHER_MODEL" \
        --output "$dir/train_teacher32.jsonl" \
        --hf-token "$HF_TOKEN" \
        --max-new-tokens 8192 \
        --load-in-8bit \
        --max-model-len 14000 \
        --gpu-memory-utilization 0.90

    python training/resample_wrong_answers.py "$dir/train_teacher32.jsonl" \
        --model-dir "$TEACHER_MODEL" \
        --output "$dir/teacher32_corrected.jsonl" \
        --hf-token "$HF_TOKEN" \
        --load-in-8bit \
        --max-model-len 14000 \
        --gpu-memory-utilization 0.90

    {
        echo "  --- stats: $dir ---"
        log_stats "$dir/train_teacher32.jsonl"     "teacher32 (initial)"
        log_stats "$dir/teacher32_corrected.jsonl" "teacher32 (resampled)"
    } | tee -a "$LOG_FILE"
    echo "$dir/train_teacher32.jsonl $dir/teacher32_corrected.jsonl" >> "$TOTAL_STATS_FILE"
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

buckets = {"teacher32 (initial)": [], "teacher32 (resampled)": []}
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
