# WildJailbreak Preprocessing (Version 2)

This script converts raw WildJailbreak data into 3 SFT tasks and 3 benchmark sets:

- `safety_combo`
- `binary_choice`
- `statement_comparison`

## Input

The script reads:

- train: `data/raw/wildjailbreak/train`
- eval: `data/raw/wildjailbreak_eval/train`

Observed raw sizes:

- train: `261559`
- eval: `2210`

## Main Transformation

For each raw example, the script adds:

- `id_temp`
- `source_split`
- `source_uid`

Then it builds three task formats.

### 1) Safety Combo

Each example becomes a 4-way classification item:

- `vanilla_benign`
- `vanilla_harmful`
- `adversarial_benign`
- `adversarial_harmful`

The question and option order are deterministically shuffled from the statement text.

### 2) Binary Choice

Each example becomes a 2-way safety question:

- `Safe`
- `Unsafe`

Mapping:

- benign -> `Safe`
- harmful -> `Unsafe`

### 3) Statement Comparison

This task pairs one harmful statement with one benign statement and asks which is safer.

Two pair types are built:

- vanilla harmful vs vanilla benign
- adversarial harmful vs adversarial benign

## Split Logic

### Train partitioning

The raw train set is first grouped by `source_uid`, shuffled, and split into 3 disjoint parts so that identical prompts stay in the same SFT train split.

Observed partition sizes:

- part A: `87186`
- part B: `87185`
- part C: `87188`

### Internal train/test split

Each partition is then split internally with:

- `internal_test_ratio = 0.05`

Observed train sizes after the 95/5 split:

- combo train source partition: `82820`
- binary train source partition: `82824`
- comparison train source partition: `40812`

### SFT outputs

From the three disjoint partitions, the script builds:

- combo train/test
- binary train/test
- comparison train/test

Observed output sizes:

1. Safety combo
	- train: `82820`
	- test: `4356`

2. Binary choice
	- train: `82824`
	- test: `4357`

3. Statement comparison
	- train: `40812`
	- test: `2148`

The small drop from the partition sizes comes from examples that do not produce valid task records for every task.

### Benchmark outputs

The benchmark is built from the separate eval set, not from the internal train/test split.

Observed benchmark sizes:

- combo: `2210`
- binary: `2210`
- comparison: `210`

## Benchmark Composition

- combo benchmark: one 4-way MCQ per eval row
- binary benchmark: one 2-way MCQ per eval row
- comparison benchmark: pairs all available benign/harmful rows, which yields `210` pairs because the eval set has `210` benign rows and `2000` harmful rows

Total benchmark size:

- `2210 + 2210 + 210 = 4630`

## Common Samples Across Train Splits

The train splits are not equal, but they overlap heavily by `source_uid`.

Using the generated train splits:

- combo train unique `source_uid`: `82820`
- binary train unique `source_uid`: `82824`
- comparison train unique `source_uid`: `40812`

Pairwise overlap of unique `source_uid` values:

- combo ∩ binary: `0`
- combo ∩ comparison: `0`
- binary ∩ comparison: `0`
- all three: `0`

Interpretation:

- the three train splits are now disjoint by `source_uid`.
- this avoids prompt-level leakage between tasks while keeping the eval-based benchmark unchanged.

## Output Paths

SFT datasets:

- `data/processed/sft/wildjailbreak/safety_combo`
- `data/processed/sft/wildjailbreak/binary_choice`
- `data/processed/sft/wildjailbreak/statement_comparison`

Benchmark datasets:

- `data/processed/benchmark/wildjailbreak/safety_combo`
- `data/processed/benchmark/wildjailbreak/binary_choice`
- `data/processed/benchmark/wildjailbreak/statement_comparison`
