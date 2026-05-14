# Multilingual component

Training and evaluation for the multilingual expert model, targeting multilingual and region-specific QA.

- Base model: `Qwen/Qwen3-1.7B`
- Target languages: Russian, Chinese, Italian, Spanish, Hindi
- English is included as a MGSM math baseline only


## Structure

```
multilingual/
├── config.py
├── generation.py
├── utils.py
├── loaders/
│   ├── ceval_loader.py
│   ├── include_loader.py
│   ├── mgsm_loader.py
│   ├── milu_loader.py
│   └── mmmlu_loader.py
├── eval/
│   ├── evaluation.py
│   ├── generate_completions.py
│   ├── run_eval.py
│   └── run_eval.sh
├── train/
│   ├── build_sft_data.py
│   ├── build_sft_data.sh
│   ├── sft.py
│   └── sft.sh
├── tests/
│   └── test_sft_data.py
├── data/
│   ├── distilled/
│   └── sft/
│       ├── per_dataset/
│       ├── mixed/
│       └── build_summary.csv
└── results/
```

Heavy data (SFT files, distilled traces, checkpoints, eval outputs) lives under `/scratch` and is not committed to Git.


## Design principles

During official CI evaluation:
- Only the system prompt is under our control
- The user turn is the raw question, unmodified
- Think mode is always enabled
- The final answer must be inside `\boxed{...}`

System prompt (defined in `config.py`):
```
You are a multilingual reasoning assistant. Think briefly before answering — avoid repetition and unnecessary elaboration. For multiple-choice questions, choose exactly one option. Always end your response with exactly one boxed answer on the last line: use \boxed{X} for multiple-choice questions, where X is the chosen letter, or \boxed{42} for numeric answers. Do not output anything after the boxed answer. Respond in the same language as the question.
```

Chat format at inference time:
```
system:    <SYSTEM_PROMPT>
user:      <raw question>
assistant: <think block> + \boxed{answer}
```


## Key constants (`config.py`)

```python
DEFAULT_MODEL_NAME = "Qwen/Qwen3-1.7B"
VAL_FRACTION = 0.15
VAL_SEED = 42

# Local eval token budgets (CI uses 16384)
DEFAULT_MAX_NEW_TOKENS_BY_TASK = {"mcq": 3072, "math": 4096}
```


## Datasets

| Dataset | HF link | Task | Languages | Split strategy |
|---|---|---|---|---|
| INCLUDE | [CohereLabs/include-base-44](https://huggingface.co/datasets/CohereLabs/include-base-44) | MCQ | RU, ZH, IT, ES, HI | 85/15 train/val from the available labeled split (`VAL_SEED=42`) |
| MGSM | [juletxara/mgsm](https://huggingface.co/datasets/juletxara/mgsm) | Math | RU, ZH, ES, EN | Same 85/15 carve-out; HI and IT skipped |
| MMMLU | [openai/MMMLU](https://huggingface.co/datasets/openai/MMMLU) | MCQ | ES, HI, IT, ZH | Subject-stratified, 4 val examples per subject; rest is train; RU not available |
| MILU | [ai4bharat/MILU](https://huggingface.co/datasets/ai4bharat/MILU) | MCQ | HI | HF `validation` → local val; HF `test` shuffled with `VAL_SEED=42` → train pool |
| C-Eval | [ceval/ceval-exam](https://huggingface.co/datasets/ceval/ceval-exam) | MCQ | ZH | Subject-stratified 80/20 of the only labeled HF `val` split; HF `test` has no labels |

RU = Russian, ZH = Chinese, IT = Italian, ES = Spanish, HI = Hindi, EN = English

All loaders return:
```python
{"id", "dataset", "language", "task_type", "prompt", "answer", "metadata"}
```

`prompt` is raw question text only — no boxed-answer instructions.

Each loader exposes `load_<dataset>_examples(..., split="train"|"val")`.



## Evaluation

### Diagnostic evaluation (`eval/run_eval.py`)

Generates one answer per example, extracts the boxed answer, and computes diagnostic metrics.

```bash
# smoke test
python -m multilingual.eval.run_eval \
    --model_name Qwen/Qwen3-1.7B \
    --n_per_language 5 \
    --output_dir multilingual/results/test \
    --save_official_format \
    --overwrite

# Full val set
python -m multilingual.eval.run_eval \
    --n_per_language full \
    --output_dir multilingual/results/base_model \
    --save_official_format

# Subset of datasets/languages
python -m multilingual.eval.run_eval \
    --datasets include mgsm \
    --languages russian chinese spanish \
    --n_per_language 50 \
    --output_dir multilingual/results/eval_small
```

Output per run:
```
results/<run_name>/
├── <dataset>_<language>.jsonl   # per-example results
├── all_results.jsonl
├── summary.csv
└── official/
    └── <dataset>_<language>.jsonl   # evaluate/score.py-compatible
```

Key metrics in `summary.csv`:
- `strict_accuracy` — main metric, matches the official boxed extractor
- `box_compliance_rate` — fraction of examples with a valid `\boxed{}`
- `valid_only_accuracy` — accuracy among examples that had a boxed answer
- `fallback_accuracy` — diagnostic only, permissive MCQ extractor
- `hit_max_tokens_rate`, `avg_generated_tokens` — generation diagnostics

### Official scoring

**Option 1:** use `--save_official_format` with `run_eval.py` (n=1, fast, use during development):
```bash
python -m evaluate.score \
    --generations multilingual/results/test/official/include_russian.jsonl \
    --benchmark multilingual
```

**Option 2:** use `generate_completions.py` (n=8, slow, use for final pass@k numbers):
```bash
python -m multilingual.eval.generate_completions \
    --model_name Qwen/Qwen3-1.7B \
    --dataset include \
    --language russian \
    --output_path multilingual/results/gens/include_russian.jsonl \
    --n 8

python -m evaluate.score \
    --generations multilingual/results/gens/include_russian.jsonl \
    --benchmark multilingual
```

The multilingual benchmark uses `method="boxed"` — the final answer is extracted from the last `\boxed{...}` in the output. MCQ: `\boxed{A}`, math: `\boxed{42}`.

Score all official files at once:
```bash
for f in multilingual/results/test/official/*.jsonl; do
    echo "=== $f ==="
    python -m evaluate.score --generations "$f" --benchmark multilingual
done
```

`HF_TOKEN` must be set. It is forwarded automatically if set in your local shell before `submit.sh`.


## SFT data (`train/build_sft_data.py`)

Builds SFT JSONL files from all loaders.

```bash
# Debug build (20 train / 10 val per dataset/language)
python -m multilingual.train.build_sft_data --debug

# Full build
python -m multilingual.train.build_sft_data

# Custom caps
python -m multilingual.train.build_sft_data \
    --max_per_language 500 --max_per_dataset 2000 --overwrite
```

Output layout:
```
data/sft/
├── per_dataset/<dataset>_<language>_<split>.jsonl   # full, no caps
├── mixed/train_mixed.jsonl                          # capped + shuffled
├── mixed/val_mixed.jsonl
└── build_summary.csv
```

Current mixed data (after deduplication):
- Train: 9212 examples (MMMLU 4000, INCLUDE 2332, C-Eval 1000, MILU 1000, MGSM 880)
- Val: 2558 examples

Deduplication is hash-based (`source_id` = hash of dataset + language + task type + prompt). Within each dataset/language, train examples whose `source_id` appears in val are removed.

SFT record format:
```python
{
    "id": "include_train_russian_0",
    "source_id": "<content hash>",
    "loader_id": "<original loader id>",
    "dataset": "include",
    "language": "russian",
    "split": "train",
    "task_type": "mcq",
    "messages": [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": "<raw question>"},
        {"role": "assistant", "content": "\\boxed{A}"},
    ],
    "answer": "A",
    "metadata": {...}
}
```

Assistant completions are currently final-answer-only (`\boxed{A}`, `\boxed{42}`). Reasoning traces will be added later via distillation.

Run unit tests before training:
```bash
pytest multilingual/tests/test_sft_data.py -v
```


## Training (`train/sft.py`)

LoRA SFT using TRL `SFTTrainer` (TRL 1.3.0).

```bash
# Debug run
python -m multilingual.train.sft \
    --run_name debug_sft \
    --max_steps 20 \
    --eval_steps 10 \
    --save_steps 10 \
    --log_steps 1 \
    --skip_wandb

# Full run
python -m multilingual.train.sft \
    --run_name sft_v1 \
    --eval_steps 250 \
    --save_steps 250 \
    --log_steps 10
```

Checkpoints saved to `multilingual/results/checkpoints/<run_name>/`. Best adapter (by eval loss) saved to `best_adapter/`.

W&B logging is on by default (project: `MNLP-Multilingual`). Pass `--skip_wandb` to disable.

### LoRA config

Defined in `train/sft.py` and mirrored in `configs/adapters/adapter_baseline.json`. All group members use the same config for merging compatibility.

```
r = 16, lora_alpha = 32, lora_dropout = 0.05
target_modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```


## Thinking strategy

Think mode is always enabled (required for model merging consistency). The intended training approach:

- Easy factual/regional MCQ: short think block (1–3 sentences)
- Harder reasoning/math: longer distilled traces from a larger Qwen3 model

Current first SFT baseline uses final-answer-only labels (`\boxed{A}`), no think block. Reasoning traces will be added as a separate ablation.

Future MCQ completion target:
```
<think>
The question asks about a specific regional fact. Option C matches because ...
</think>
\boxed{C}
```


## Known limitations

- Evaluation uses HuggingFace Transformers directly, not vLLM
- Local val splits are carved from public test/val data — results are local development metrics, not official scores
- Reasoning traces not yet generated; current SFT uses final-answer-only labels
- MGSM translated examples may be noisy; use for reasoning diagnostics only, filter carefully before training
- Fallback accuracy is diagnostic only, do not report as main result
