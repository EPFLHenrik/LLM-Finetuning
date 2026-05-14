# Multilingual component

This folder contains the multilingual evaluation code and the planned training pipeline for the multilingual expert model.

The goal is to improve the model on multilingual and region-specific question answering, while keeping the evaluation format compatible with the official MNLP CI evaluation.

Current base model:

```
Qwen/Qwen3-1.7B
```

Target languages:

```
Russian, Chinese, Italian, Spanish, Hindi
```

English is also included for MGSM/math sanity checks, but it is not a target language.

---

## 1. Main design principles

The multilingual component follows the official CI evaluation contract as closely as possible.

During evaluation:

1. The system prompt is controlled by us.
2. The user turn is the raw question only.
3. No extra per-question instructions are injected into the user prompt.
4. Qwen3 think mode is always enabled.
5. The final answer should be written inside `\boxed{...}`.

The system prompt is defined in:

```
multilingual/config.py
```

Current prompt:

```
You are a multilingual reasoning assistant. Think carefully before answering. Always enclose your final answer in \boxed{}, for example \boxed{A} for multiple-choice questions or \boxed{42} for numeric answers. Respond in the same language as the question.
```

The model therefore receives prompts in this chat format:

```
system: multilingual reasoning + boxed-answer instruction
user: raw dataset question
assistant: generated answer with think block and boxed final answer
```

The user turn must stay raw because this matches the official CI setting.

---

## 2. Repository structure

```
multilingual/
├── config.py
├── generation.py
├── utils.py
├── loaders/
│   ├── include_loader.py
│   ├── mgsm_loader.py
│   ├── mmmlu_loader.py
│   ├── milu_loader.py
│   └── ceval_loader.py
├── eval/
│   ├── evaluation.py
│   ├── run_eval.py
│   ├── generate_completions.py
│   └── run_eval.sh
└── results/
```

---

## 3. Main files

### `config.py`

Defines global constants:

```
REPO
DEFAULT_MODEL_NAME = "Qwen/Qwen3-1.7B"
TARGET_LANGUAGES
SYSTEM_PROMPT
DEFAULT_MAX_NEW_TOKENS_BY_TASK
VAL_FRACTION = 0.15
VAL_SEED = 42
```


Local evaluation uses smaller token budgets than the official CI:

```
mcq: 2048
math: 4096
```

The official CI uses a larger generation budget, currently 16384 tokens.

### `generation.py`

Handles model loading and generation.

Main functions:

```
build_messages(raw_question)
load_model_and_tokenizer(model_name, hf_token)
generate_answer(...)
```

Generation uses Qwen3 chat formatting with thinking enabled:

```
tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)
```

By default, generation is greedy:

```
do_sample = False
```

Sampling can be enabled with:

```
--do_sample --temperature 0.6 --top_p 0.95 --top_k 20
```

### `utils.py`

Contains common helpers for:

```
dataset split selection
deterministic train/val splitting
answer normalization
JSONL saving
```

The most important function is:

```
get_train_val_split(ds, split_name)
```

It creates deterministic local train/validation splits when a dataset does not already provide separate train and validation splits.

---

## 4. Dataset loaders

All loaders return examples in the same internal format:

```
{
    "id": "...",
    "dataset": "...",
    "language": "...",
    "task_type": "mcq" or "math",
    "prompt": "...",
    "answer": "...",
    "metadata": {...}
}
```

The `prompt` field is always the raw question text. It does not include boxed-answer instructions.

| Dataset | Loader | HF dataset | Task | Languages | Local split strategy | Notes |
|---|---|---|---|---|---|---|
| INCLUDE | `include_loader.py` | `CohereLabs/include-base-44` | MCQ | RU, ZH, IT, ES, HI | deterministic train/val split with `VAL_SEED=42`, `VAL_FRACTION=0.15` | Main regional multilingual QA dataset. English skipped. |
| MGSM | `mgsm_loader.py` | `juletxara/mgsm` | Math | RU, ZH, ES, EN | deterministic train/val split from the only labeled split | Used for multilingual mathematical reasoning. HI and IT skipped. |
| MMMLU | `mmmlu_loader.py` | `openai/MMMLU` | MCQ | ES, HI, IT, ZH | stratified local train/val split by subject | Broad multilingual knowledge. Russian not available. |
| MILU | `milu_loader.py` | `ai4bharat/MILU` | MCQ | HI | validation split for local val; test split used as train pool | Hindi local/exam-style QA. |
| C-Eval | `ceval_loader.py` | `ceval/ceval-exam` | MCQ | ZH | labeled `val` split split into 80% train / 20% val, stratified by subject | Chinese exam QA. Official test has no labels. |

Language abbreviations:

```
RU = Russian
ZH = Chinese
IT = Italian
ES = Spanish
HI = Hindi
EN = English
```

Each loader exposes a `load_<dataset>_examples(..., split="train" or "val")` function and, when useful, a `load_<dataset>_splits()` convenience function.

## 5. Evaluation pipeline

There are two evaluation modes.

---

## 5.1 Diagnostic local evaluation

Main file:

```
multilingual/eval/run_eval.py
```

This script:

1. Loads a model.
2. Loads validation examples from selected datasets/languages.
3. Generates one answer per example.
4. Extracts the boxed answer.
5. Computes diagnostic metrics.
6. Saves detailed outputs.

Example command:

```
python -m multilingual.eval.run_eval \
    --model_name Qwen/Qwen3-1.7B \
    --n_per_language 5 \
    --output_dir multilingual/results/test \
    --save_official_format \
    --overwrite
```

By default, it evaluates on:

```
include, mgsm, mmmlu, milu, ceval
```

and on all configured languages.

You can restrict datasets and languages:

```
python -m multilingual.eval.run_eval \
    --datasets include mgsm \
    --languages russian chinese spanish \
    --n_per_language 50 \
    --output_dir multilingual/results/eval_small \
    --overwrite
```

You can run the full local validation split with:

```
python -m multilingual.eval.run_eval \
    --n_per_language full \
    --output_dir multilingual/results/eval_full \
    --overwrite
```

Outputs:

```
multilingual/results/<run_name>/
├── include_russian.jsonl
├── include_chinese.jsonl
├── ...
├── all_results.jsonl
├── summary.csv
└── official/
    ├── include_russian.jsonl
    ├── ...
```

The per-example JSONL files contain:

```
id
dataset
language
task_type
gold
prediction
correct
missing_box
fallback_prediction
fallback_correct
n_generated_tokens
hit_max_tokens
max_new_tokens
prompt
output
metadata
```

The summary file contains also:

- `strict_accuracy`: accuracy using only the official boxed-answer extraction.
- `box_compliance_rate`: fraction of examples where the model produced a boxed answer.
- `valid_only_accuracy`: accuracy among examples with a valid boxed answer.
- `fallback_accuracy`: diagnostic MCQ accuracy using a more permissive fallback extractor.
- `hit_max_tokens_rate`: fraction of generations that reached the token limit.
- `avg_generated_tokens`: average number of generated tokens.

The most important metric is `strict_accuracy`, because it best matches the official boxed evaluation.
The fallback metric is only diagnostic. It helps identify whether the model knew the answer but failed the required output format.

---

## 5.2 Official-format scoring

There are two ways to produce files compatible with `evaluate/score.py`.

### Option 1: use `run_eval.py --save_official_format`

Example:

```
python -m multilingual.eval.run_eval \
    --model_name Qwen/Qwen3-1.7B \
    --n_per_language 5 \
    --output_dir multilingual/results/test \
    --save_official_format \
    --overwrite
```

This creates files in:

```
multilingual/results/test/official/
```

Each row has the schema:

```
{
    "prompt": "...",
    "answer": "...",
    "completions": ["..."]
}
```

Then score with:

```
python -m evaluate.score \
    --generations multilingual/results/test/official/include_russian.jsonl \
    --benchmark multilingual
```

### Option 2: use `generate_completions.py`

Main file:

```
multilingual/eval/generate_completions.py
```

This script generates `n` completions per question. It is useful for pass@k evaluation.

Example:

```
python -m multilingual.eval.generate_completions \
    --model_name Qwen/Qwen3-1.7B \
    --dataset include \
    --language russian \
    --output_path multilingual/results/gens/include_russian.jsonl \
    --n 8
```

Then score with:

```
python -m evaluate.score \
    --generations multilingual/results/gens/include_russian.jsonl \
    --benchmark multilingual
```

The official scoring path uses:

```
evaluate/score.py
evaluate/benchmarks.py
evaluate/extract_answer.py
evaluate/pass_at_k.py
```

For the multilingual benchmark, `evaluate.score` uses the `boxed` method:

```
BENCHMARK_TO_METHOD["multilingual"] = "boxed"
```

This means the final answer must be extractable from the last `\boxed{...}` in the generated output.

---

## 6. Answer extraction

The local diagnostic evaluator imports:

```
extract_boxed_answer
is_equiv
```

from:

```
evaluate/extract_answer.py
```

For MCQ tasks:

```
prediction == gold letter
```

after normalization.

For math tasks:

```
is_equiv(prediction, gold)
```

is used to compare normalized math answers.

The official scoring path uses:

```
evaluate/benchmarks.py
```

For `benchmark="multilingual"`, it extracts answers with:

```
extract_boxed_answer(text, strip_double_curly_brace=True)
```

Therefore, the expected final answer format is `\boxed{A}` for MCQ and `\boxed{42}` for numeric/math questions. 


---

## 7. Running the test script

The current shell script is:

```
multilingual/eval/run_eval.sh
```

It runs a small test evaluation:

```
python -m multilingual.eval.run_eval \
    --model_name Qwen/Qwen3-1.7B \
    --n_per_language 5 \
    --output_dir multilingual/results/test \
    --save_official_format \
    --overwrite
```

Then it scores each official-format JSONL file:

```
for f in multilingual/results/test/official/*.jsonl; do
    echo "=== $f ==="
    python -m evaluate.score --generations "$f" --benchmark multilingual
done
```

Before running evaluation, we need to make sure `HF_TOKEN` is set, which should normally be the case as long as we have it set locally on our local shell before we submitted the `submit.sh` script, which transmits is as one of the environment variables in the pod. Otherwise:

```
export HF_TOKEN=hf_xxx
```

---

## 8. Current adapter baseline

Current adapter baseline (LoRA) is stored in:

```
configs/adapters/adapter_baseline.json
```

Current LoRA settings:

```
r = 16
lora_alpha = 32
lora_dropout = 0.05
bias = none
task_type = CAUSAL_LM
```

Target modules:

```
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

For model merging, all group members should use the same LoRA rank, alpha, dropout, and target modules, otherwise merging adapters becomes harder and less reliable.

---

## 9. Planned training pipeline

Training is not implemented yet in this folder.
Recommended future structure:

```
multilingual/
├── train/
│   ├── build_sft_data.py
│   ├── train_lora.py
│   └── run_train.sh
└── data/
    ├── train/
    └── val/
```

The training pipeline should reuse the existing dataset loaders.

Planned steps:

1. Load local train splits from the dataset loaders.
2. Convert each example into SFT chat format.
3. Add assistant completions with a `think` block and a boxed final answer.
4. Train a LoRA/QLoRA adapter on top of `Qwen/Qwen3-1.7B`.
5. Evaluate checkpoints with the existing evaluation pipeline.
6. Select the best checkpoint using local validation performance.
7. Save the final adapter for later merging with the other group experts.

---

## 10. Training data format

The SFT examples should follow the same inference-time structure:

```
system: SYSTEM_PROMPT
user: raw question
assistant: <think>...</think>\n\boxed{...}
```

For MCQ examples, the final answer should be:

```
\boxed{A}
\boxed{B}
\boxed{C}
\boxed{D}
```

For math examples, the final answer should be:

```
\boxed{42}
```

or the corresponding normalized numeric answer.

Important:

The training data should always include the boxed final answer. Otherwise the model may learn to answer correctly but fail the official extractor. Because think mode is enabled during inference, the SFT completions should also include a `think` block. Otherwise the model may be trained in a format that conflicts with the generation setup.

---

## 11. Thinking strategy

Think mode is always enabled for this component.

The intended training strategy is to teach the model a spectrum of reasoning depth:

1. Easy factual or regional MCQ questions:
   - use a short, non-empty think block
   - usually 1–3 sentences

2. Harder reasoning/math questions:
   - use longer reasoning traces
   - preferably distilled from a larger Qwen3 model or taken from datasets that already contain reasoning

Example MCQ completion:

```
<think>
The question asks about a specific regional fact. Option C matches the relevant fact, while the other options describe unrelated alternatives.
</think>
\boxed{C}
```

Example math completion:

```
<think>
Work through the arithmetic carefully and derive the requested quantity step by step.
</think>
\boxed{42}
```

Avoid training many examples with completely empty think blocks, because this may teach the model to skip thinking entirely. A short but non-empty think block is a safer middle ground.



## 12. Useful commands

Small test evaluation:

```
python -m multilingual.eval.run_eval \
    --model_name Qwen/Qwen3-1.7B \
    --n_per_language 5 \
    --output_dir multilingual/results/test \
    --save_official_format \
    --overwrite
```

Evaluate only INCLUDE:

```
python -m multilingual.eval.run_eval \
    --datasets include \
    --languages russian chinese italian spanish hindi \
    --n_per_language 50 \
    --output_dir multilingual/results/include_50 \
    --save_official_format \
    --overwrite
```

Evaluate only MGSM:

```
python -m multilingual.eval.run_eval \
    --datasets mgsm \
    --languages russian chinese spanish english \
    --n_per_language full \
    --output_dir multilingual/results/mgsm_full \
    --save_official_format \
    --overwrite
```

Generate 8 completions for pass@k scoring:

```
python -m multilingual.eval.generate_completions \
    --model_name Qwen/Qwen3-1.7B \
    --dataset include \
    --language russian \
    --output_path multilingual/results/gens/include_russian.jsonl \
    --n 8
```

Score official-format generations:

```
python -m evaluate.score \
    --generations multilingual/results/gens/include_russian.jsonl \
    --benchmark multilingual
```

---

## 13. Known limitations and TODOs

Current limitations:

1. Training is not implemented yet.
2. Dataset mixing and sampling weights are not yet defined.
3. Reasoning traces for SFT are not yet generated.
4. QLoRA training scripts are not yet added.
5. Evaluation currently loads Hugging Face models directly with Transformers, not vLLM.
6. Some datasets use project-local splits from official test/validation data, so results should be interpreted as local development metrics, not official benchmark scores.
7. Fallback accuracy is only diagnostic and should not be reported as the main result.

Next steps:

1. Add `multilingual/train/build_sft_data.py`.
2. Export unified SFT train/val JSONL files.
3. Generate short think traces for MCQ data.
4. Generate or collect longer reasoning traces for MGSM.
5. Add `multilingual/train/train_lora.py`.
6. Train a first LoRA adapter.
7. Evaluate the adapter with `run_eval.py`.
8. Compare against the base model.
9. Save the best adapter for merging.