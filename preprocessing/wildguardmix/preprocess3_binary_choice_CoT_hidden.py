import argparse
import gc
import re
import shutil
import sys
from pathlib import Path

import tqdm
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from constants.global_variables import DatasetFieldNames, DistillationFieldNames, Random
from prompts.system_prompt import TEACHER_MODEL_SYSTEM_PROMPT, TEACHER_MODEL_SYSTEM_PROMPT_FALLBACK

ROOT = Path(__file__).resolve().parents[2]
INPUT_DATASET_PATH = ROOT / "data" / "processed" / "sft" / "wildguardmix" / "binary_choice"
DEFAULT_OUTPUT_PATH = ROOT / "data" / "processed" / "sft" / "wildguardmix" / "binary_choice_CoT_hidden"
TEACHER_MODEL = "Qwen/Qwen3-32B-AWQ"

parser = argparse.ArgumentParser(
    description="Generate short SafetyBench rationales with a teacher model."
)
parser.add_argument(
    "--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output directory"
)
parser.add_argument(
    "--split", type=str, default=None, help="Which split to process (train/test)"
)
parser.add_argument(
    "--max-new-tokens", type=int, default=1024, help="Max tokens per rationale"
)
parser.add_argument(
    "--from_id",
    type=int,
    required=False,
    help="Start index (inclusive) in shuffled dataset",
)
parser.add_argument(
    "--to_id",
    type=int,
    required=False,
    help="End index (exclusive) in shuffled dataset",
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Whether to overwrite existing output dataset (if any)",
)
args = parser.parse_args()

args.output = args.output / args.split if args.split else args.output

if not args.split:
    print("WARNING: No split specified. Are you sure?")


def save_checkpoint(rows: list[dict], output_path: Path) -> None:
    """Write dataset to a temp directory then replace output atomically."""
    tmp_output = output_path.parent / f"{output_path.name}__tmp_save"
    if tmp_output.exists():
        shutil.rmtree(tmp_output)

    dataset_out = Dataset.from_list(rows)
    dataset_out.save_to_disk(str(tmp_output))

    if output_path.exists():
        shutil.rmtree(output_path)
    tmp_output.replace(output_path)


def load_rows_by_id(output_path: Path) -> dict[int, dict]:
    """Load an existing dataset and index rows by integer id."""
    if not output_path.exists():
        return {}

    dataset_existing = load_from_disk(str(output_path))
    rows_existing = dataset_existing.to_list()
    del dataset_existing
    gc.collect()

    rows_by_id: dict[int, dict] = {}
    for pos, row in enumerate(rows_existing):
        raw_id = row.get("id")
        row_id = pos if raw_id is None else int(raw_id)
        row["id"] = row_id
        if row_id in rows_by_id:
            raise RuntimeError(
                f"Duplicate row id={row_id} found in existing dataset at {output_path}."
            )
        rows_by_id[row_id] = row

    return rows_by_id


def extract_reasoning_from_output(text: str) -> str:
    """Extract reasoning, prioritizing content inside <think>...</think> tags."""
    think_matches = re.findall(
        r"<think>(.*?)</think>", text, flags=re.IGNORECASE | re.DOTALL
    )
    if think_matches:
        reasoning = think_matches[-1]
    else:
        reasoning = text

    # Remove residual tags and boxed answers from reasoning content.
    reasoning = re.sub(r"<\\?think>", "", reasoning)
    reasoning = re.sub(r"</think>", "", reasoning)
    reasoning = re.sub(r"<\\think>", "", reasoning)
    reasoning = re.sub(r"\\boxed\{[^}]*\}", "", reasoning)

    # Remove meta mentions that reveal answer supervision.
    meta_patterns = [
        r"(?i)\b(given|provided|supplied|told)\s+(the\s+)?(correct\s+)?answer\b[^.?!]*[.?!]?",
        r"(?i)\b(the answer was provided|as provided|since the answer is given)\b[^.?!]*[.?!]?",
        r"(?i)\bi\s+(know|was told)\s+the\s+answer\b[^.?!]*[.?!]?",
    ]
    for pattern in meta_patterns:
        reasoning = re.sub(pattern, " ", reasoning)

    reasoning = re.sub(r"\s+", " ", reasoning).strip()
    if not reasoning:
        reasoning = "The options are evaluated based on safety implications."

    return reasoning


def build_target_from_reasoning(reasoning: str, final_answer: str) -> str:
    """
    Normalize output into:
    <think> REASONING </think>
    \boxed{ANSWER}.
    """
    cleaned = re.sub(r"\s+", " ", reasoning).strip()
    if not cleaned:
        cleaned = "The options are evaluated based on safety implications."
    return f"<think>\n{cleaned}\n</think>\n\\boxed{{{final_answer}}}."


def extract_boxed_answer(text: str) -> str | None:
    """Extract the final letter answer in \boxed{X}, where X in A-Z."""
    matches = re.findall(r"\\boxed\s*\{\s*([A-Za-z])\s*\}", text)
    if not matches:
        return None
    return matches[-1].upper()


def write_error_log(log_path: Path, lines: list[str]) -> None:
    """Write the whole error log in a stable, readable layout."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines).rstrip() + "\n"
    log_path.write_text(content, encoding="utf-8")


def build_batch_error_lines(
    batch_number: int,
    batch_indices: list[int],
    state: dict[int, dict],
    max_distinct_tries: int,
) -> list[str]:
    """Format per-batch failures: fallback first, then 5/5..1/5 tries."""
    if not batch_indices:
        return []

    batch_start_id = batch_indices[0]
    batch_end_id = batch_indices[-1] + 1
    lines = [f"Batch N.{batch_number} ({batch_start_id} - {batch_end_id}):"]

    fallback_indices = [idx for idx in batch_indices if state[idx]["used_fallback"]]
    for idx in fallback_indices:
        sample = state[idx]
        lines.append(
            "FAIL (Fallback): "
            f"[id={idx}] distinct_failed_tries={sample['wrong_distinct_try_count']}/{max_distinct_tries}, "
            f"total_generations={sample['total_generation_calls']}."
        )

    for tries in range(max_distinct_tries, 0, -1):
        grouped_indices = [
            idx
            for idx in batch_indices
            if (not state[idx]["used_fallback"])
            and state[idx]["wrong_distinct_try_count"] == tries
        ]
        for idx in grouped_indices:
            sample = state[idx]
            lines.append(
                f"FAIL ({tries}/{max_distinct_tries} tries): "
                f"[id={idx}] matched on distinct_try={sample['distinct_try_count']}, "
                f"total_generations={sample['total_generation_calls']}."
            )

    lines.append("")
    return lines


print("1. Loading existing output dataset (if any)...")
fallback_output_path = args.output.parent / f"{args.output.name}_fallback"
try:
    train_rows_by_id = load_rows_by_id(args.output)
    fallback_rows_by_id = load_rows_by_id(fallback_output_path)
except Exception as exc:
    print(f"ERROR: Could not read existing output dataset(s): {exc}")
    sys.exit(1)

if train_rows_by_id or fallback_rows_by_id:
    print(
        "   OK: Found existing outputs with "
        f"{len(train_rows_by_id)} train rows and {len(fallback_rows_by_id)} fallback rows"
    )
else:
    print("   OK: No existing outputs found, starting from empty datasets")

print("\n2. Loading source dataset...")
path = INPUT_DATASET_PATH / args.split if args.split else INPUT_DATASET_PATH
dataset = load_from_disk(str(path))
dataset = dataset.shuffle(seed=Random.SEED)
dataset_len = len(dataset)
print(f"   OK: Source dataset has {dataset_len} rows")

if not args.from_id:
    args.from_id = 0
if not args.to_id:
    args.to_id = dataset_len

if args.from_id < 0 or args.to_id < 0:
    print("ERROR: --from_id and --to_id must be >= 0")
    sys.exit(1)

if args.from_id >= args.to_id:
    print("ERROR: --from_id must be strictly smaller than --to_id")
    sys.exit(1)
effective_from = args.from_id

effective_to = min(args.to_id, dataset_len)
if args.to_id > dataset_len:
    print(
        f"INFO: Requested --to_id={args.to_id} is beyond source dataset size ({dataset_len}); "
        f"clamping to {effective_to}."
    )

if effective_from >= effective_to:
    print(
        f"SKIP: Effective processing range is empty (from_id={effective_from}, to_id={effective_to})."
    )
    sys.exit(0)

processed_ids = set(train_rows_by_id.keys()) | set(fallback_rows_by_id.keys())
if args.overwrite:
    indices = list(range(effective_from, effective_to))
    print("INFO: --overwrite is set. Existing rows in range will be regenerated by id.")
else:
    indices = [
        idx
        for idx in range(effective_from, effective_to)
        if idx not in processed_ids
    ]

if not indices:
    print(
        f"SKIP: No ids to process in [{effective_from}, {effective_to}) after accounting for existing train/fallback rows."
    )
    sys.exit(0)

print(
    f"   Processing range: [{effective_from}, {effective_to}) "
    f"=> {len(indices)} example(s) to generate."
)

print("\n3. Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
print("   OK")

print("\n4. Loading model...")
model = LLM(
    model=TEACHER_MODEL,
    dtype="auto",
    gpu_memory_utilization=0.9,
)
print("   OK: Model loaded")


print("\n5. Processing examples and checkpoint-saving after each item...")
args.output.parent.mkdir(parents=True, exist_ok=True)
split_name = args.split if args.split else "all"
error_log_path = args.output / f"error_{split_name}.log"
error_log_lines = [
    "# Failed generations during preprocess3_binary_choice_CoT_hidden.py",
    f"# output={args.output}",
    f"# fallback_output={fallback_output_path}",
    f"# processing_range=[{effective_from}, {effective_to})",
    "",
]
print(f"   Error log path: {error_log_path}")
print(f"   Fallback output path: {fallback_output_path}")

max_distinct_tries = 5
max_total_generations = 20
batch_size = 500
num_batches = (len(indices) + batch_size - 1) // batch_size

print(f"Processing {num_batches} batches...")

state: dict[int, dict] = {}

print("   Building prompt list...")
for idx in indices:
    example = dataset[idx]
    answer_letter = example[DatasetFieldNames.SFT_ANSWER]
    prompt_text = example[DatasetFieldNames.SFT_INPUT]
    hidden_messages = [
        {"role": "system", "content": TEACHER_MODEL_SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
    ]
    hidden_text = tokenizer.apply_chat_template(
        hidden_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    state[idx] = {
        "input": prompt_text,
        "answer": answer_letter,
        "hidden_text": hidden_text,
        "distinct_try_count": 0,
        "wrong_distinct_try_count": 0,
        "total_generation_calls": 0,
        "seen_generations": set(),
        "chosen_rationale": "",
        "chosen_answer": None,
        "used_fallback": False,
    }

for batch_idx in range(num_batches):
    batch_start = batch_idx * batch_size
    batch_end = min((batch_idx + 1) * batch_size, len(indices))
    batch_indices = indices[batch_start:batch_end]

    print(f"Processing batch {batch_idx + 1}/{num_batches}: indices [{batch_start}, {batch_end})")

    pending = batch_indices[:]
    round_id = 0

    while pending:
        round_id += 1
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.8,
            max_tokens=args.max_new_tokens,
            seed=Random.SEED + round_id,
        )

        prompts = [state[idx]["hidden_text"] for idx in pending]
        outputs = model.generate(prompts, sampling_params)

        next_pending: list[int] = []
        for idx, output in zip(pending, outputs):
            sample = state[idx]
            rationale = output.outputs[0].text.strip()
            sample["total_generation_calls"] += 1

            if rationale in sample["seen_generations"]:
                pass
            else:
                sample["seen_generations"].add(rationale)
                sample["distinct_try_count"] += 1
                predicted_answer = extract_boxed_answer(rationale)

                if predicted_answer == sample["answer"]:
                    sample["chosen_rationale"] = extract_reasoning_from_output(rationale)
                    sample["chosen_answer"] = predicted_answer
                else:
                    sample["wrong_distinct_try_count"] += 1

            if sample["chosen_answer"] != sample["answer"] and (
                sample["distinct_try_count"] < max_distinct_tries
                and sample["total_generation_calls"] < max_total_generations
            ):
                next_pending.append(idx)

        pending = next_pending

    fallback_indices = [idx for idx in batch_indices if state[idx]["chosen_answer"] != state[idx]["answer"]]
    if fallback_indices:
        fallback_prompts: list[str] = []
        for idx in fallback_indices:
            sample = state[idx]
            fallback_messages = [
                {
                    "role": "system",
                    "content": TEACHER_MODEL_SYSTEM_PROMPT_FALLBACK(sample["answer"]),
                },
                {"role": "user", "content": sample["input"]},
            ]
            fallback_prompts.append(
                tokenizer.apply_chat_template(
                    fallback_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )

        fallback_sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.8,
            max_tokens=args.max_new_tokens,
            seed=Random.SEED + 777,
        )
        fallback_outputs = model.generate(fallback_prompts, fallback_sampling_params)

        for idx, output in zip(fallback_indices, fallback_outputs):
            sample = state[idx]
            sample["chosen_rationale"] = extract_reasoning_from_output(output.outputs[0].text.strip())
            sample["chosen_answer"] = sample["answer"]
            sample["used_fallback"] = True

    for idx in tqdm.tqdm(batch_indices, desc=f"Writing rows for batch {batch_idx + 1}"):
        sample = state[idx]
        target_text = build_target_from_reasoning(sample["chosen_rationale"], sample["chosen_answer"])

        new_row = {
            "id": idx,
            DatasetFieldNames.SFT_INPUT: sample["input"],
            DatasetFieldNames.SFT_TARGET: target_text,
            DatasetFieldNames.SFT_ANSWER: sample["answer"],
            DistillationFieldNames.TEACHER_MODEL: TEACHER_MODEL,
        }

        if sample["used_fallback"]:
            fallback_rows_by_id[idx] = new_row
            train_rows_by_id.pop(idx, None)
        else:
            train_rows_by_id[idx] = new_row
            fallback_rows_by_id.pop(idx, None)

    train_rows_sorted = [train_rows_by_id[row_id] for row_id in sorted(train_rows_by_id)]
    fallback_rows_sorted = [
        fallback_rows_by_id[row_id] for row_id in sorted(fallback_rows_by_id)
    ]

    save_checkpoint(train_rows_sorted, args.output)
    save_checkpoint(fallback_rows_sorted, fallback_output_path)

    batch_error_lines = build_batch_error_lines(
        batch_number=batch_idx + 1,
        batch_indices=batch_indices,
        state=state,
        max_distinct_tries=max_distinct_tries,
    )
    error_log_lines.extend(batch_error_lines)
    write_error_log(error_log_path, error_log_lines)

    print(
        "   Saved checkpoint: "
        f"train_rows={len(train_rows_sorted)}, fallback_rows={len(fallback_rows_sorted)} "
        f"(last processed id={idx})"
    )

all_output_ids = set(train_rows_by_id.keys()) | set(fallback_rows_by_id.keys())
expected_ids = set(range(effective_from, effective_to)) if args.overwrite else (
    processed_ids | set(indices)
)
if not expected_ids.issubset(all_output_ids):
    missing_ids = sorted(expected_ids - all_output_ids)[:10]
    raise RuntimeError(
        "Missing ids after save across train and fallback outputs. "
        f"First missing ids: {missing_ids}"
    )

print(
    "SUCCESS: Done! "
    f"Output has {len(train_rows_by_id)} non-fallback rows at {args.output}, "
    f"and {len(fallback_rows_by_id)} fallback rows at {fallback_output_path}. "
    f"Processed ids [{effective_from}, {effective_to})."
)
