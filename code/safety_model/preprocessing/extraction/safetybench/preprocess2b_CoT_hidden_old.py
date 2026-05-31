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
from constants.system_prompt import SYSTEM_PROMPT_MCQ, TEACHER_MODEL_SYSTEM_PROMPT, TEACHER_MODEL_SYSTEM_PROMPT_FALLBACK

ROOT = Path(__file__).resolve().parents[2]
INPUT_DATASET_PATH = (
    ROOT / "data" / "processed" / "sft" / "safetybench" / "only_answers"
)
DEFAULT_OUTPUT_PATH = ROOT / "data" / "processed" / "sft" / "safetybench" / "CoT_hidden"
TEACHER_MODEL_PATH = str(
    ROOT.parent.parent.parent
    / "hf_cache"
    / "hub"
    / "models--Qwen--Qwen3-8B"
    / "snapshots"
    / "b968826d9c46dd6066d109eabc6255188de91218"
)

TEACHER_MODEL = "Qwen/Qwen3-32B-AWQ"

parser = argparse.ArgumentParser(
    description="Generate short SafetyBench rationales with a teacher model."
)
parser.add_argument(
    "--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output directory"
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


def extract_reasoning_from_output(text: str) -> str:
    """Extract reasoning, prioritizing content inside ◣...◢ tags."""
    think_matches = re.findall(
        r"◣(.*?)◢", text, flags=re.IGNORECASE | re.DOTALL
    )
    if think_matches:
        reasoning = think_matches[-1]
    else:
        reasoning = text

    # Remove residual tags and boxed answers from reasoning content.
    reasoning = re.sub(r"<\\?think>", "", reasoning)
    reasoning = re.sub(r"◣", "", reasoning)
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
    <think> 
    REASONING 
    </think>
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
existing_rows = []
existing_count = 0
if args.output.exists():
    try:
        existing_dataset = load_from_disk(str(args.output))
        existing_count = len(existing_dataset)
        existing_rows = existing_dataset.to_list()
        dataset_len = len(existing_dataset)
        del existing_dataset
        gc.collect()
        print(f"   OK: Found existing output with {existing_count} rows")
    except Exception as exc:
        print(f"ERROR: Could not read existing output dataset at {args.output}: {exc}")
        sys.exit(1)
else:
    print("   OK: No existing output found, starting from empty dataset")

print("\n2. Loading source dataset...")
dataset = load_from_disk(str(INPUT_DATASET_PATH))
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


if args.from_id > existing_count:
    print(
        f"SKIP: --from_id ({args.from_id}) is greater than output row count ({existing_count}). "
        "Impossible to run, this will corrupt the order of the output dataset compared to the input dataset."
        "Aborting."
    )
    sys.exit(0)

if args.to_id <= existing_count and not args.overwrite:
    print(
        f"SKIP: --to_id ({args.to_id}) is smaller than or equal to output row count ({existing_count}). "
        "Nothing to run."
    )
    sys.exit(0)

effective_from = (
    max(args.from_id, existing_count) if not args.overwrite else args.from_id
)
if effective_from != args.from_id and not args.overwrite:
    print(
        f"INFO: Resuming from output row count. Requested --from_id={args.from_id}, "
        f"but starting at {effective_from}."
    )
if args.overwrite:
    print(
        f"INFO: --overwrite is set. Starting from --from_id={args.from_id} and ignoring existing output row count ({existing_count}). "
        "This will lead to duplicate rows in the output dataset if --from_id < existing row count."
    )

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

print(
    f"   Processing range: [{effective_from}, {effective_to}) "
    f"=> {effective_to - effective_from} example(s)."
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
split_name = "all"
error_log_path = args.output / f"error_{split_name}.log"
error_log_lines = [
    "# Failed generations during preprocess2b_CoT_hidden.py",
    f"# output={args.output}",
    f"# processing_range=[{effective_from}, {effective_to})",
    "",
]
print(f"   Error log path: {error_log_path}")

max_distinct_tries = 5
max_total_generations = 20
indices = list(range(effective_from, effective_to))
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
            DatasetFieldNames.SFT_INPUT: sample["input"],
            DatasetFieldNames.SFT_TARGET: target_text,
            DatasetFieldNames.SFT_ANSWER: sample["answer"],
            DistillationFieldNames.TEACHER_MODEL: TEACHER_MODEL,
        }

        if idx < len(existing_rows):
            existing_rows[idx] = new_row
        elif idx == len(existing_rows):
            existing_rows.append(new_row)
        else:
            raise RuntimeError(
                "Non-contiguous write detected: cannot write source index "
                f"{idx} when output has only {len(existing_rows)} rows. "
                "Run earlier indices first to keep ordering consistent."
            )

    save_checkpoint(existing_rows, args.output)

    batch_error_lines = build_batch_error_lines(
        batch_number=batch_idx + 1,
        batch_indices=batch_indices,
        state=state,
        max_distinct_tries=max_distinct_tries,
    )
    error_log_lines.extend(batch_error_lines)
    write_error_log(error_log_path, error_log_lines)

    print(
        f"   Saved checkpoint: {len(existing_rows)} total rows "
        f"(last processed id={idx})"
    )

print(
    f"SUCCESS: Done! Output now has {len(existing_rows)} rows at {args.output}. "
    f"Processed ids [{effective_from}, {effective_to})."
)
