#!/usr/bin/env bash
# Preprocess all datasets in <project-root>/data/datasets into <project-root>/data/preprocessing.
# Run from any directory; paths are resolved relative to this script's location.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GK_DIR="$(dirname "$SCRIPT_DIR")"            # general_knowledge_model/
DATA_DIR="$GK_DIR/../../data"                # project-root/data/
DATASETS_DIR="$DATA_DIR/datasets"
OUT_DIR="$DATA_DIR/preprocessing"
PY="$GK_DIR/preprocessing"

step() { echo; echo "=== $* ==="; }

# ─── TRAIN DATASETS ───────────────────────────────────────────────────────────

step "MATH — merge config subfolders into flat train/test splits"
python "$PY/merge_math_splits.py" \
  --base-dir "$DATASETS_DIR/train_datasets/MATH"

step "MATH — convert to 4-choice MCQ format"
python "$PY/preprocess_math_mcq.py" \
  "$DATASETS_DIR/train_datasets/MATH/train.jsonl" \
  "$OUT_DIR/train_datasets/MATH/train.jsonl" \
  --seed 42

step "FinTrain — convert to MCQ format"
python "$PY/convert_fintrain_to_mcq.py" \
  "$DATASETS_DIR/train_datasets/FinTrain/train.jsonl" \
  "$OUT_DIR/train_datasets/FinTrain/train.jsonl"

step "MMLU auxiliary train"
python "$PY/preprocess_mmlu_auxiliary_train.py" \
  "$DATASETS_DIR/train_datasets/mmlu_auxtrain_labelled/default/train.jsonl" \
  --output "$OUT_DIR/train_datasets/mmlu_auxtrain/train.jsonl"

step "MMLU auxiliary train — filter to selected categories"
python "$PY/filter_dataset_by_categories.py" \
  --input  "$OUT_DIR/train_datasets/mmlu_auxtrain/train.jsonl" \
  --output "$OUT_DIR/train_datasets/mmlu_auxtrain/train_filtered.jsonl"

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
cat "$OUT_DIR/train_datasets"/*/train.jsonl > "$OUT_DIR/full_train.jsonl"

echo
echo "All preprocessing complete. Outputs written to: $OUT_DIR"
