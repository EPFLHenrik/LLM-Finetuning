#!/usr/bin/env bash
# Preprocess all datasets in <project-root>/data/general_knowledge/raw into <project-root>/data/general_knowledge/processed.
# Run from any directory; paths are resolved relative to this script's location.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GK_DIR="$(dirname "$SCRIPT_DIR")"               # general_knowledge_model/
PROJECT_DIR="$(dirname "$(dirname "$GK_DIR")")"  # project root
DATA_DIR="$PROJECT_DIR/data/general_knowledge"
DATASETS_DIR="$DATA_DIR/raw"
OUT_DIR="$DATA_DIR/processed"
PY="$GK_DIR/preprocessing"

HF_TOKEN="${HF_TOKEN:?HF_TOKEN environment variable is required}"

step() { echo; echo "=== $* ==="; }

# ─── DOWNLOAD ─────────────────────────────────────────────────────────────────

step "Download dataset from HuggingFace"
python - <<PYEOF
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="cs-552-2026-mystery-machine/general_knowledge",
    repo_type="dataset",
    local_dir="$DATA_DIR",
    token=os.environ["HF_TOKEN"],
)
print("Download complete.")
PYEOF
# HF repo uses the singular form; normalise to the plural form used throughout
[ -f "$DATA_DIR/processed/train_datasets/merged_distillation.jsonl" ] && \
    mv "$DATA_DIR/processed/train_datasets/merged_distillation.jsonl" \
       "$DATA_DIR/processed/train_datasets/merged_distillations.jsonl"

step "Download student model (Qwen/Qwen3-1.7B)"
python - <<PYEOF
import os
from pathlib import Path
from huggingface_hub import snapshot_download
output_dir = Path("$PROJECT_DIR/checkpoints/general_knowledge_model/qwen_thinking3")
if (output_dir / "config.json").exists():
    print("Already present, skipping.")
else:
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id="Qwen/Qwen3-1.7B", local_dir=str(output_dir), token=os.environ["HF_TOKEN"])
    print("Download complete.")
PYEOF

step "Download teacher model (Qwen/Qwen3-14B)"
python - <<PYEOF
import os
from pathlib import Path
from huggingface_hub import snapshot_download
output_dir = Path("$PROJECT_DIR/checkpoints/general_knowledge_model/teacher")
if (output_dir / "config.json").exists():
    print("Already present, skipping.")
else:
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id="Qwen/Qwen3-14B", local_dir=str(output_dir), token=os.environ["HF_TOKEN"])
    print("Download complete.")
PYEOF

# ─── TRAIN DATASETS ───────────────────────────────────────────────────────────

step "MATH — merge config subfolders into flat train/test splits"
python "$PY/merge_math_splits.py" \
  --base-dir "$DATASETS_DIR/train_datasets/MATH"

step "MATH — convert to 4-choice MCQ format"
python "$PY/preprocess_math_mcq.py" \
  "$DATASETS_DIR/train_datasets/MATH/train.jsonl" \
  --output "$OUT_DIR/train_datasets/MATH/train.jsonl" \
  --seed 42

step "FinTrain — convert to MCQ format"
python "$PY/convert_fintrain_to_mcq.py" \
  "$DATASETS_DIR/train_datasets/FinTrain/train.jsonl" \
  "$OUT_DIR/train_datasets/FinTrain/train.jsonl"

step "MMLU auxiliary train — filter to selected categories"
python "$PY/filter_dataset_by_categories.py" \
  --input  "$DATASETS_DIR/train_datasets/mmlu_auxtrain_labelled/default/train.jsonl" \
  --output "$DATASETS_DIR/train_datasets/mmlu_auxtrain_labelled/default/train_filtered.jsonl"

step "MMLU auxiliary train — preprocess filtered split"
python "$PY/preprocess_mmlu_auxiliary_train.py" \
  "$DATASETS_DIR/train_datasets/mmlu_auxtrain_labelled/default/train_filtered.jsonl" \
  --output "$OUT_DIR/train_datasets/mmlu_aux_filtered/train.jsonl"

step "CommonsenseQA — train + validation"
python "$PY/preprocess_hf.py" \
  --dataset-path  tau/commonsense_qa \
  --input-dir     "$DATASETS_DIR/train_datasets/commonsense_qa" \
  --output-dir    "$OUT_DIR/train_datasets/commonsense_qa" \
  --question-keys question \
  --choice-keys   choices \
  --answer-keys   answerKey

step "AI2 ARC Challenge — train + validation"
python "$PY/preprocess_hf.py" \
  --dataset-path   allenai/ai2_arc \
  --dataset-config ARC-Challenge \
  --input-dir      "$DATASETS_DIR/train_datasets/ai2_arc_challenge" \
  --output-dir     "$OUT_DIR/train_datasets/ai2_arc_challenge" \
  --question-keys  question \
  --choice-keys    choices \
  --answer-keys    answerKey

step "AI2 ARC Easy — train + validation"
python "$PY/preprocess_hf.py" \
  --dataset-path   allenai/ai2_arc \
  --dataset-config ARC-Easy \
  --input-dir      "$DATASETS_DIR/train_datasets/ai2_arc_easy" \
  --output-dir     "$OUT_DIR/train_datasets/ai2_arc_easy" \
  --question-keys  question \
  --choice-keys    choices \
  --answer-keys    answerKey

step "ScienceQA — train (drops image-only rows)"
python "$PY/preprocess_scienceqa.py" \
  --input  "$DATASETS_DIR/train_datasets/science_qa/train.jsonl" \
  --output "$OUT_DIR/train_datasets/science_qa/train.jsonl"

step "ScienceQA — validation (drops image-only rows)"
python "$PY/preprocess_scienceqa.py" \
  --input  "$DATASETS_DIR/train_datasets/science_qa/validation.jsonl" \
  --output "$OUT_DIR/train_datasets/science_qa/validation.jsonl"

step "SFT Econ — train (MCQ rows only)"
python "$PY/preprocess_sft_econ.py" \
  "$DATASETS_DIR/train_datasets/sft_dataset_econ/train.jsonl" \
  "$OUT_DIR/train_datasets/sft_dataset_econ/train.jsonl"

# ─── BENCHMARKS ───────────────────────────────────────────────────────────────

step "MMLU (cais) — all subject test splits"
python "$PY/preprocess_cais_mmlu.py" \
  --input-dir  "$DATASETS_DIR/benchmarks/cais__mmlu" \
  --output-dir "$OUT_DIR/benchmarks/cais_mmlu"

step "MMLU-Pro — test split"
python "$PY/preprocess_mmlu_pro_test.py" \
  --input  "$DATASETS_DIR/benchmarks/MMLU-pro/test.jsonl" \
  --output "$OUT_DIR/benchmarks/MMLU-pro/test.jsonl"

step "MMLU-Pro — validation split"
python "$PY/preprocess_mmlu_pro_test.py" \
  --input  "$DATASETS_DIR/benchmarks/MMLU-pro/validation.jsonl" \
  --output "$OUT_DIR/benchmarks/MMLU-pro/validation.jsonl"

step "CommonsenseQA benchmark — test + validation"
python "$PY/preprocess_hf.py" \
  --dataset-path  tau/commonsense_qa \
  --input-dir     "$DATASETS_DIR/benchmarks/commonsense_qa" \
  --output-dir    "$OUT_DIR/benchmarks/commonsense_qa" \
  --question-keys question \
  --choice-keys   choices \
  --answer-keys   answerKey

step "AI2 ARC Challenge benchmark"
python "$PY/preprocess_hf.py" \
  --dataset-path   allenai/ai2_arc \
  --dataset-config ARC-Challenge \
  --input-dir      "$DATASETS_DIR/benchmarks/ai2_arc_challenge" \
  --output-dir     "$OUT_DIR/benchmarks/ai2_arc_challenge" \
  --question-keys  question \
  --choice-keys    choices \
  --answer-keys    answerKey

step "AI2 ARC Easy benchmark"
python "$PY/preprocess_hf.py" \
  --dataset-path   allenai/ai2_arc \
  --dataset-config ARC-Easy \
  --input-dir      "$DATASETS_DIR/benchmarks/ai2_arc_easy" \
  --output-dir     "$OUT_DIR/benchmarks/ai2_arc_easy" \
  --question-keys  question \
  --choice-keys    choices \
  --answer-keys    answerKey

step "ScienceQA benchmark — test (drops image-only rows)"
python "$PY/preprocess_scienceqa.py" \
  --input  "$DATASETS_DIR/benchmarks/science_qa/test.jsonl" \
  --output "$OUT_DIR/benchmarks/science_qa/test.jsonl"

step "SFT Econ benchmark — test (MCQ rows only)"
python "$PY/preprocess_sft_econ.py" \
  "$DATASETS_DIR/benchmarks/sft_dataset_econ/test.jsonl" \
  "$OUT_DIR/benchmarks/sft_dataset_econ/test.jsonl"

step "Merge all train splits into full_train.jsonl"
cat "$OUT_DIR/train_datasets"/*/train.jsonl > "$OUT_DIR/train_datasets/full_train.jsonl"

echo
echo "All preprocessing complete. Outputs written to: $OUT_DIR"
