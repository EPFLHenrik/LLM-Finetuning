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
from prompts.system_prompt import SYSTEM_PROMPT_MCQ, TEACHER_MODEL_SYSTEM_PROMPT

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


def append_failure_log(log_path: Path, message: str) -> None:
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(message + "\n")


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
failed_log_path = args.output.parent / "failed.log"
failed_log_path.write_text(
    (
        "# Failed answer attempts during preprocess2b_CoT_hidden.py\n"
        f"# output={args.output}\n"
        f"# processing_range=[{effective_from}, {effective_to})\n"
    ),
    encoding="utf-8",
)
print(f"   Failure log path: {failed_log_path}")

max_distinct_tries = 5
max_total_generations = 20

for idx in tqdm.tqdm(range(effective_from, effective_to), desc="Processing examples"):
    example = dataset[idx]
    # answer_letter = example[DatasetFieldNames.SFT_ANSWER] # GOOD LINE, but ds not updated
    answer_letter = example["expected_boxed_answer"]

    hidden_messages = [
        {"role": "system", "content": TEACHER_MODEL_SYSTEM_PROMPT},
        {"role": "user", "content": example[DatasetFieldNames.SFT_INPUT]},
    ]
    hidden_text = tokenizer.apply_chat_template(
        hidden_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    distinct_try_count = 0
    total_generation_calls = 0
    seen_generations: set[str] = set()
    chosen_rationale = ""
    chosen_answer = None

    while (
        distinct_try_count < max_distinct_tries
        and total_generation_calls < max_total_generations
    ):
        # Recommended params for thinking mode at https://huggingface.co/Qwen/Qwen3-1.7B
        sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            max_tokens=args.max_new_tokens,
            seed=idx * 1000 + total_generation_calls + 1,
        )

        outputs = model.generate([hidden_text], sampling_params)
        rationale = outputs[0].outputs[0].text.strip()
        total_generation_calls += 1

        # Duplicate generations are ignored so retries are actually diverse.
        if rationale in seen_generations:
            message = (
                f"[id={idx}] Duplicate generation ignored "
                f"(call={total_generation_calls}, distinct={distinct_try_count})."
            )
            print(f"FAIL: {message}")
            append_failure_log(failed_log_path, f"FAIL: {message}")
            continue

        seen_generations.add(rationale)
        distinct_try_count += 1

        predicted_answer = extract_boxed_answer(rationale)
        if predicted_answer == answer_letter:
            chosen_rationale = extract_reasoning_from_output(rationale)
            chosen_answer = predicted_answer
            break

        preview = rationale.replace("\n", " ")[:220]
        message = (
            f"[id={idx}] Wrong answer on distinct_try={distinct_try_count}/"
            f"{max_distinct_tries}: predicted={predicted_answer!r}, expected={answer_letter!r}. "
            f"Preview: {preview}"
        )
        print(f"FAIL: {message}")
        append_failure_log(failed_log_path, f"FAIL: {message}")

    if chosen_answer != answer_letter:
        fallback_prompt = f"""{example[DatasetFieldNames.SFT_INPUT]}

            Use this target answer: {answer_letter}.
            Reason to that answer naturally and do not mention that any target answer was provided.
            """

        fallback_messages = [
            {"role": "system", "content": TEACHER_MODEL_SYSTEM_PROMPT},
            {"role": "user", "content": fallback_prompt},
        ]
        fallback_text = tokenizer.apply_chat_template(
            fallback_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        fallback_sampling_params = SamplingParams(
            temperature=0.2,
            top_p=0.95,
            max_tokens=args.max_new_tokens,
            seed=idx * 1000 + 777,
        )
        fallback_outputs = model.generate([fallback_text], fallback_sampling_params)
        chosen_rationale = extract_reasoning_from_output(
            fallback_outputs[0].outputs[0].text.strip()
        )
        chosen_answer = answer_letter
        fallback_message = (
            f"[id={idx}] Fallback used after {distinct_try_count} distinct failed tries "
            f"({total_generation_calls} total generations)."
        )
        print(f"FAIL: {fallback_message}")
        append_failure_log(failed_log_path, f"FAIL: {fallback_message}")

    target_text = build_target_from_reasoning(chosen_rationale, chosen_answer)

    new_row = {
        DatasetFieldNames.SFT_INPUT: example[DatasetFieldNames.SFT_INPUT],
        DatasetFieldNames.SFT_TARGET: target_text,
        DatasetFieldNames.SFT_ANSWER: answer_letter,
        DistillationFieldNames.TEACHER_MODEL: TEACHER_MODEL,
    }

    if idx < len(existing_rows):
        # Update existing row (in case of overwrite or resuming with from_id < existing_count)
        existing_rows[idx] = new_row
    else:
        existing_rows.append(new_row)

    if (idx + 1) % 10 == 0 or idx == effective_to - 1:
        save_checkpoint(existing_rows, args.output)
        print(
            f"   Saved checkpoint: {len(existing_rows)} total rows (last processed id={idx})"
        )

print(
    f"SUCCESS: Done! Output now has {len(existing_rows)} rows at {args.output}. "
    f"Processed ids [{effective_from}, {effective_to})."
)
