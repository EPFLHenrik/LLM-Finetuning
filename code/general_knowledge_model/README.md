# general_knowledge_model

Self-contained code for training and evaluating a general-knowledge MCQ model.
All internal cross-package imports point back into this directory; external dependencies are third-party packages only (transformers, trl, peft, vllm, wandb, etc.).

---

## evaluate/

| File | What it does |
|---|---|
| `extract_answer.py` | Core answer extraction utilities: extracts content from `\boxed{}`, checks equivalence, normalises final answers. Ported from OpenCompass. |
| `benchmarks.py` | Per-benchmark answer extraction and correctness check. Wraps `extract_answer.py` with three methods — `boxed`, `exact`, and `knowledge` (letter-choice or free-text). |

---

## preprocessing/

Scripts that convert raw dataset formats into the `prompt`/`answer` JSONL schema expected by training scripts, or into chat-style JSONL for SFT.

| File | What it does |
|---|---|
| `convert_fintrain_to_mcq.py` | Converts a financial training dataset into standardised MCQ JSONL; filters out records that reference figures/exhibits or have truncated choices. |
| `convert_swissai_to_messages.py` | Converts `swissai_merged.jsonl` flat rows into chat-message format (system / user / assistant) wrapping completions as `<think>…</think>\boxed{answer}`. |
| `merge_math_splits.py` | Concatenates MATH config subfolders (`algebra/`, `geometry/`, …) into flat `train.jsonl` / `test.jsonl`. |
| `preprocess_cais_mmlu.py` | Converts cais/mmlu test splits to `prompt`/`answer` JSONL; maps numeric answer indices to letters. |
| `preprocess_hf.py` | Generic HuggingFace dataset preprocessor: builds prompts from configurable question/choice/answer field names. |
| `preprocess_math_mcq.py` | Converts MATH `train.jsonl` into 4-choice MCQ format; extracts the gold `\boxed{}` answer and generates type-appropriate distractors. |
| `preprocess_mmlu_auxiliary_train.py` | Preprocesses MMLU `auxiliary_train` into `prompt`/`answer` JSONL; handles both index-based and letter-based answer formats. |
| `preprocess_mmlu_pro_test.py` | Preprocesses MMLU-Pro test JSONL into `prompt`/`answer` format with lettered options appended. |
| `preprocess_scienceqa.py` | Preprocesses ScienceQA train split; drops image-based rows, formats choices, and converts the answer to a letter. |
| `preprocess_sft_econ.py` | Preprocesses COURSEMO/sft_dataset_econ into MCQ JSONL; drops image rows and extracts the answer letter from the reasoning field. |
| `filter_dataset_by_categories.py` | Keeps only JSONL rows whose `task` field matches a configured category list. |

---

## scripts/

Utilities for data management, model setup, filtering, evaluation, and deployment.

| File | What it does |
|---|---|
| `download_hf.py` | Downloads a HuggingFace dataset and writes each split as JSONL. |
| `download_teacher_model.py` | Downloads a HuggingFace model snapshot (default: `Qwen/Qwen3-14B-Instruct`) into `models/teacher`. |
| `ensure_boxed_targets.py` | Converts flat `prompt`/`answer` JSONL into chat-style JSONL where the assistant message is `\boxed{answer}`; run before LoRA SFT. |
| `evaluate_training.py` | Evaluates every merged checkpoint in a folder; runs vLLM inference on benchmark datasets and writes per-checkpoint JSON accuracy reports. |
| `filter_partial_success.py` | From multiple `generate_answers.py` runs, keeps prompts answered correctly 1–2 times out of N — the optimal training signal for GRPO. |
| `filter_remove_incorrect.py` | Removes rows with `correct == false` from a JSONL file in-place, saving a `.bak` backup. |
| `model_config.py` | Loads a checkpoint, applies in-file config edits (system message, generation config, chat template), and saves a new checkpoint directory. |
| `push_model.py` | Uploads a local model directory to `cs-552-2026-mystery-machine/general_knowledge_model` on HuggingFace. |
| `run_validation.py` | Runs local validation over `validation_samples/*.jsonl` using a local checkpoint; falls back to `gpt2` for quick pipeline testing. |
| `setup_baseline.py` | Downloads and configures a baseline checkpoint (default: `Qwen/Qwen3-1.7B`); patches the system message and chat template. |
| `summarize_benchmarks.py` | Aggregates per-benchmark `evaluation_summary.json` files into a single JSONL summary. |
| `task_field_distribution.py` | Prints the value distribution of a field (default: `task`) in a JSONL dataset; useful for inspecting data before preprocessing. |
| `build_dpo_dataset.py` | Builds DPO training pairs from student run files and teacher distillation data; pairs each prompt where the student was wrong with the teacher's correct completion as chosen and the student's wrong completion as rejected. |
| `merge_lora.py` | Merges a LoRA adapter into a base model by applying weights manually (W_new = W + scaling × B @ A) without requiring bitsandbytes/CUDA; saves the merged model to disk. |


---

## training/

Core training and inference scripts.

| File | What it does |
|---|---|
| `generate_answers.py` | Uses the teacher model (via vLLM) to generate completions for each prompt; scores each answer and writes a JSONL with a `correct` flag. |
| `grpo_train.py` | GRPO training with TRL's `GRPOTrainer`; uses exact-match boxed-answer reward; saves a merged checkpoint at the end. |
| `lora_post_train.py` | LoRA SFT training with TRL's `SFTTrainer` on chat-formatted JSONL (produced by the `preprocess_*_chat.py` scripts). |
| `merge_reasoning_from_student_teacher.py` | Joins two JSONL files (student and teacher runs) on matching prompt+answer pairs and outputs chat-style rows. |
| `resample_wrong_answers.py` | Re-runs vLLM inference only on rows with `correct == false` and rescores them; correct rows pass through unchanged. |
| `dpo_train.py` | DPO training on `(prompt, chosen, rejected)` pairs using TRL's `DPOTrainer`; supports LoRA and W&B logging. Input produced by `scripts/build_dpo_dataset.py`. |

---

## Setup

### Data

`1_run_preprocessing.sh` downloads the dataset from HuggingFace automatically (requires `HF_TOKEN`). The repository contains both `raw/` (used by the full pipeline) and `processed/` (pre-computed `full_train.jsonl`, `merged_distillations.jsonl`, and `wrong_distillations.jsonl`). To skip preprocessing and start directly from `3_sft.sh`, run `1_run_preprocessing.sh` for the download step only and then proceed to script 3.

### Models

Both models are downloaded automatically by `1_run_preprocessing.sh` (requires `HF_TOKEN`). Downloads are skipped if the target directory already contains a `config.json`.

| Model | HuggingFace source | Local path |
|---|---|---|
| Student base model (Qwen3-1.7B) | `Qwen/Qwen3-1.7B` | `$PROJECT_DIR/checkpoints/general_knowledge_model/qwen_thinking3` |
| Teacher model (Qwen3-14B) | `Qwen/Qwen3-14B` | `$PROJECT_DIR/checkpoints/general_knowledge_model/teacher` |

---

## How to Reproduce the best Model
### in the /shell-scripts folder

End-to-end pipeline scripts. Run them in numbered order from any working directory; all paths are resolved relative to the script's location.

| Script | What it does |
|---|---|
| `1_run_preprocessing.sh` | Converts raw datasets under `data/general_knowledge/raw/` into `prompt`/`answer` JSONL under `data/general_knowledge/processed/train_datasets/` and `data/general_knowledge/processed/benchmarks/`. Concatenates all train splits into `data/general_knowledge/processed/full_train.jsonl`. |
| `2_distill_all.sh` | For each preprocessed train dataset, runs the student and teacher models via `generate_answers.py`, resamples wrong answers with `resample_wrong_answers.py`, and merges the corrected outputs with `merge_reasoning_from_student_teacher.py`. Appends merged rows to `data/general_knowledge/processed/merged_distillations.jsonl` and logs per-dataset accuracy stats. Requires `HF_TOKEN`. |
| `3_sft.sh` | Applies the custom system-message chat template to the base model in-place via `model_config.py --in-place`, then runs LoRA SFT on `data/general_knowledge/processed/merged_distillations.jsonl`; saves a merged checkpoint to `checkpoints/general_knowledge_model/full_sft`. Requires `WANDB_API_KEY`. |
| `4_prepare_grpo.sh` | Runs three seeded inference passes over `data/general_knowledge/processed/full_train.jsonl` using `checkpoints/general_knowledge_model/full_sft`, then filters to the partial-success zone (correct 1–2 out of 3 times) via `filter_partial_success.py`, writing `data/general_knowledge/processed/grpo_runs/partial_success.jsonl`. Requires `HF_TOKEN`. |
| `5_run_grpo.sh` | GRPO fine-tuning on `data/general_knowledge/processed/grpo_runs/partial_success.jsonl` starting from `checkpoints/general_knowledge_model/full_sft`; saves the final model to `checkpoints/general_knowledge_model/sft_grpo`. |
| `6_prepare_dpo.sh` | Builds DPO training pairs via `scripts/build_dpo_dataset.py` using the three GRPO run files and `data/general_knowledge/processed/merged_distillations.jsonl` as the teacher source; writes `data/general_knowledge/processed/dpo_pairs.jsonl`. |
| `7_run_gpo.sh` | DPO fine-tuning on `data/general_knowledge/processed/dpo_pairs.jsonl` starting from `checkpoints/general_knowledge_model/sft_grpo`; saves the final model to `checkpoints/general_knowledge_model/sft_grpo_dpo`. Requires `WANDB_API_KEY`. |
| `8_run_all_benchmark_evals.sh` | Evaluates `checkpoints/general_knowledge_model/sft_grpo_dpo` against all preprocessed benchmark splits under `data/general_knowledge/processed/benchmarks/`; writes per-benchmark JSON reports to `results/sft_grpo_dpo/benchmarks/` and a combined summary to `results/sft_grpo_dpo/benchmarks/benchmark_summary.json`. |
