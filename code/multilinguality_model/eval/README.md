# Multilingual evaluation

This folder contains two evaluation utilities:

1. vLLM-based evaluation for running multilingual benchmark inference.
2. Standalone scoring for already-generated completions.

## vLLM evaluation

Run model inference and scoring on a JSONL evaluation file.

```
python -m multilinguality_model.eval.run_eval_vllm \
  --model_name Qwen/Qwen3-1.7B \
  --val_file data/multilinguality/sft/debug_test/val_mixed.jsonl \
  --output_dir runs/benchmark/results/multilinguality_model/debug_qwen3_base \
  --n 5 \
  --overwrite \
  --tensor_parallel_size 1
```

The evaluator writes:

```
val_results.jsonl
provided_results.jsonl
all_results.jsonl
summary_by_dataset_language.csv
summary_by_language.csv
summary_by_dataset.csv
summary.txt
run_metadata.json
```

The exact set of files depends on whether validation samples, provided samples, or both are evaluated.

## Combined result summaries

After one or more evaluation runs, aggregate completed result folders into one combined JSON and CSV summary:

```
python -m multilinguality_model.eval.aggregate_results
```

This writes:

```
runs/benchmark/results/multilinguality_model/combined_results.json
runs/benchmark/results/multilinguality_model/combined_results.csv
```

The combined summary contains one row per model / benchmark / dataset / language group.

## Standalone scorer

The standalone scorer evaluates already-generated completions. It does not run a model and does not load datasets.

Use it when you already have a JSONL file with one row per problem:

```
{"prompt": "What is 2+2?", "answer": "4", "completions": ["...\\boxed{4}", "...\\boxed{5}"]}
```

Required fields:

| Field | Required | Notes |
|---|---|---|
| `answer` | yes | Gold reference. `reference` is accepted as a synonym. |
| `completions` | yes | Non-empty list of model outputs. All rows must have the same number of completions. |
| `prompt` | no | Echoed in detailed output if present. |

Run:

```
python -m multilinguality_model.eval.score \
  --generations my_multilingual_gens.jsonl \
  --benchmark multilingual
```

Output example:

```
pass@1=0.4500, pass@8=0.7200 (n_problems=10, n_completions=8, method=boxed)
```

Add `--output scored.json` to write detailed per-problem scoring information.

## Benchmark scoring methods

| `--benchmark` | Extraction method | Notes |
|---|---|---|
| `math` | `boxed` | Last `\boxed{...}` only; math equivalence on the inner string. |
| `knowledge` | `knowledge` | Uses boxed extraction and text fallback. For single-letter gold answers, extracts a choice label. |
| `multilingual` | `boxed` | Same extraction method as math. |
| `safety` | `boxed` | Same extraction method as math. |

The `--method {boxed|knowledge|exact}` argument can override the benchmark default.

## Files

```
run_eval_vllm.py      vLLM inference and evaluation runner
eval_utils.py         summary helpers used by run_eval_vllm.py
aggregate_results.py  combined JSON/CSV summary builder
score.py              standalone scorer for existing generations
benchmarks.py         answer extraction methods for standalone scoring
pass_at_k.py          pass@k estimator
extract_answer.py     boxed-answer and math-equivalence helpers
evaluation.py         older/general pandas evaluator
```

## Saved results
 
Pre-computed evaluation results for three model checkpoints are stored in `eval/results/`:
 
```
eval/results/
  Qwen3-1.7B/     base model (Qwen/Qwen3-1.7B)
  v1/             best SFT multilingual checkpoint
  final-merged/   final individual model (TIES merge of v1 + Russian/Hindi specialists)
```
 
Each folder contains three subfolders: `full/` (all benchmark datasets), `mmlu_prox_hard/`, and `validation_set_mnlp/`, the validation set of 10 examples provided to us, used as complementary evaluation only..