# CS-552 Multilingual Regional MCQA Benchmark v1

Custom-made evaluation benchmark for multilingual regional knowledge MCQA across 5 languages. 2500 examples total, 500 per language.

**Note:** all benchmark examples are held out and never used for training.

## Structure

```
benchmark_all.jsonl          # v1 benchmark (2500 examples)
by_language/                 # partitioned by language
by_dataset/                  # partitioned by dataset × language
small/                       # 500-example subset for fast eval
  benchmark_all_small.jsonl
  by_language/
mmlu_prox/                   # separate MMLU-ProX hard benchmark
  mmlu_prox_hard_1000.jsonl  # combined hard benchmark (1000 examples)
  by_language/               # 200 examples per language
```

## Composition

Each language combines INCLUDE (regional knowledge baseline) with Global MMLU (translated general knowledge) and a language-specific native dataset.

| Language | INCLUDE | Global MMLU | Native | Total |
|----------|---------|-------------|--------|-------|
| Russian  | 100     | 200         | 200 MERA | 500 |
| Chinese  | 100     | 100         | 150 C-Eval + 150 CMMLU | 500 |
| Hindi    | 100     | 200         | 200 MILU | 500 |
| Italian  | 100     | 200         | 200 EXAMS | 500 |
| Spanish  | 100     | 200         | 200 HEAD-QA | 500 |

Dataset totals: INCLUDE 500 · Global MMLU 900 · MERA 200 · C-Eval 150 · CMMLU 150 · MILU 200 · EXAMS 200 · HEAD-QA 200.

## Subject coverage

| Language | Dataset | Subjects |
|----------|---------|----------|
| Chinese | C-Eval | 52 |
| Chinese | CMMLU | 67 |
| Chinese | Global MMLU | 57 |
| Chinese | INCLUDE | 26 |
| Hindi | Global MMLU | 57 |
| Hindi | INCLUDE | 15 |
| Hindi | MILU | 41 |
| Italian | EXAMS | 12 |
| Italian | Global MMLU | 57 |
| Italian | INCLUDE | 17 |
| Russian | Global MMLU | 57 |
| Russian | INCLUDE | 13 |
| Russian | MERA | 21 |
| Spanish | Global MMLU | 57 |
| Spanish | HEAD-QA | 6 |
| Spanish | INCLUDE | 9 |

## Answer distribution

Most examples are 4-option (A–D). HEAD-QA includes a small number of 5-option questions (E).

| Language | A | B | C | D | E |
|----------|---|---|---|---|---|
| Chinese  | 123 | 121 | 114 | 142 | 0 |
| Hindi    | 129 | 130 | 132 | 109 | 0 |
| Italian  | 119 | 145 | 116 | 120 | 0 |
| Russian  | 138 | 151 |  99 | 112 | 0 |
| Spanish  | 112 | 126 | 147 | 106 | 9 |

## Schema

Each record is a JSON object with fields:

| Field | Description |
|-------|-------------|
| `source_id` | Original dataset identifier (used for contamination checks) |
| `language` | Language name (russian, chinese, hindi, italian, spanish) |
| `dataset` | Source dataset name |
| `prompt` | Question text with answer options |
| `answer` | Gold answer letter (A–E) |
| `id` | Record identifier |
| `split` | Always `eval` |
| `task_type` | Always `mcq` |
| `metadata` | Dataset-specific fields (subject, domain, etc.) |

## Small benchmark

`small/` contains a 500-example subset (100 per language) preserving the same proportional composition as the full benchmark. Can be used for fast tests.

| Language | INCLUDE | Global MMLU | Native | Total |
|----------|---------|-------------|--------|-------|
| Russian  | 20 | 40 | 40 MERA | 100 |
| Chinese  | 20 | 20 | 30 C-Eval + 30 CMMLU | 100 |
| Hindi    | 20 | 40 | 40 MILU | 100 |
| Italian  | 20 | 40 | 40 EXAMS | 100 |
| Spanish  | 20 | 40 | 40 HEAD-QA | 100 |


## MMLU-ProX hard benchmark

In addition to the v1 benchmark, we provide a separate hard benchmark based on MMLU-ProX.

MMLU-ProX is kept separate from `benchmark_all.jsonl` so that v1 scores remain comparable across model runs. The hard benchmark contains 1000 examples total: 200 examples per language across the same five languages.

The examples are sampled only from the held-out MMLU-ProX eval split and are never used for training. MMLU-ProX is a multilingual translation/adaptation of the English MMLU-Pro benchmark, so many examples are parallel across languages. To make cross-language comparison fair, we sample 200 shared underlying `base_question_id` values and include their versions in all five languages. This ensures that differences in score across languages are less affected by differences in question difficulty.

Sampling is mildly weighted toward regional, cultural, society-oriented, health, and biology-related subjects, using the same subject weights as the MMLU-ProX 5k training subset.

| File | Description | Total |
|------|-------------|-------|
| `mmlu_prox/mmlu_prox_hard_1000.jsonl` | Combined MMLU-ProX hard benchmark | 1000 |
| `mmlu_prox/by_language/mmlu_prox_russian_hard_200.jsonl` | Russian MMLU-ProX hard split | 200 |
| `mmlu_prox/by_language/mmlu_prox_chinese_hard_200.jsonl` | Chinese MMLU-ProX hard split | 200 |
| `mmlu_prox/by_language/mmlu_prox_hindi_hard_200.jsonl` | Hindi MMLU-ProX hard split | 200 |
| `mmlu_prox/by_language/mmlu_prox_italian_hard_200.jsonl` | Italian MMLU-ProX hard split | 200 |
| `mmlu_prox/by_language/mmlu_prox_spanish_hard_200.jsonl` | Spanish MMLU-ProX hard split | 200 |

MMLU-ProX may contain variable numbers of answer options, up to A-J.