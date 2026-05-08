
from pathlib import Path
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer
import re
import argparse
import tqdm
import sys
import shutil
import gc
import random
from constants.global_variables import DatasetFieldNames, DistillationFieldNames
from vllm import LLM, SamplingParams

ROOT = Path(__file__).resolve().parents[2]
INPUT_DATASET_PATH = ROOT / "data" / "processed" / "sft" / "wildguardmix" / "subcategory_mcq_answers_only"
DEFAULT_OUTPUT_PATH = ROOT / "data" / "processed" / "sft" / "wildguardmix" / "subcategory_CoT"
TEACHER_MODEL_PATH = str(ROOT.parent.parent.parent / "hf_cache" / "hub" / "models--Qwen--Qwen3-8B" / "snapshots" / "b968826d9c46dd6066d109eabc6255188de91218")

parser = argparse.ArgumentParser(
    description="Generate short Subcategory rationales with a teacher model."
)
parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output directory")
parser.add_argument("--split", type=str, default=None, help="Which split to process (train/test)")
parser.add_argument("--max-new-tokens", type=int, default=1024, help="Max tokens per rationale")
parser.add_argument("--from_id", type=int, required=False, help="Start index (inclusive) in shuffled dataset")
parser.add_argument("--to_id", type=int, required=False, help="End index (exclusive) in shuffled dataset")
parser.add_argument("--overwrite", action="store_true", help="Whether to overwrite existing output dataset (if any)")
args = parser.parse_args()

args.output = args.output / args.split if args.split else args.output

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
path = INPUT_DATASET_PATH / args.split if args.split else INPUT_DATASET_PATH
dataset = load_from_disk(str(path))
dataset = dataset.shuffle(seed=42)
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

effective_from = max(args.from_id, existing_count) if not args.overwrite else args.from_id
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
tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
print("   OK")

print("\n4. Loading model...")
model = LLM(
    model=TEACHER_MODEL_PATH,
    dtype="auto",
    gpu_memory_utilization=0.9,
)
print(f"   OK: Model loaded")

conclusion_sentences = [
    "Therefore, the correct answer is",
    "Thus, the answer is",
    "In conclusion, the correct choice is",
    "Hence, the answer is",
    "As a result, the correct answer is",
    "Consequently, the answer is",
    "To sum up, the correct answer is",
]


def build_target_from_model_output(text: str, expected_answer: str) -> str:
    """
    Normalize model output into:
    <think> CLEANED_COT </think> Random conclusion \boxed{ANSWER}
    """
    # Remove any think-tag variants produced by the model.
    cleaned = re.sub(r"<\\?think>", "", text)
    cleaned = re.sub(r"</think>", "", cleaned)
    cleaned = re.sub(r"<\\think>", "", cleaned)

    # Remove any boxed answers generated by the model; we append the expected one.
    cleaned = re.sub(r"\\boxed\{[^}]*\}", "", cleaned)

    # Collapse spacing/newlines for compact, stable CoTs.
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        cleaned = "The options are evaluated based on safety implications."

    return f"<think>\n{cleaned}\n</think>\n\\boxed{{{expected_answer}}}."

print("\n5. Processing examples and checkpoint-saving after each item...")
args.output.parent.mkdir(parents=True, exist_ok=True)

for idx in tqdm.tqdm(range(effective_from, effective_to), desc="Processing examples"):
    example = dataset[idx]
    answer_letter = re.search(r"\\boxed\{([A-E])\}", example["target"]).group(1)

    teacher_prompt = f"""You are creating high-quality training data for multiple-choice safety reasoning.
        Question:
        {example['prompt']}

        Correct answer: {answer_letter}

        Write a complete and optimal chain of thought for why this answer is correct.

        Rules:
        - Keep the reasoning complete but not excessively long.
        - If the question isn't trivial or looks adversarial, don't hesitate to think for a bit longer while remaining efficient.
        - Use only information present in the question and answer choices.
        - Do not mention that the answer was provided.
        - Do not restate the full question.
        - Do not use meta commentary, hidden reasoning, or generic filler.
        - Keep the reasoning specific and concrete.
        """
    
    messages = [{"role": "user", "content": teacher_prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
    )
    
    outputs = model.generate([text], sampling_params)
    rationale = outputs[0].outputs[0].text.strip()

    # Normalize to the required target format with explicit think tags and boxed answer.
    target_text = build_target_from_model_output(rationale, answer_letter)

    new_row = {
        DatasetFieldNames.SFT_INPUT: example["prompt"],
        DatasetFieldNames.SFT_TARGET: target_text,
        DatasetFieldNames.SFT_ANSWER: answer_letter,
        DistillationFieldNames.TEACHER_MODEL: TEACHER_MODEL_PATH,
    }
    
    if idx < len(existing_rows):
        # Update existing row (in case of overwrite or resuming with from_id < existing_count)
        existing_rows[idx] = new_row
    else:
        existing_rows.append(new_row)

    if (idx + 1) % 10 == 0 or idx == effective_to - 1:
        save_checkpoint(existing_rows, args.output)
        print(f"   Saved checkpoint: {len(existing_rows)} total rows (last processed id={idx})")

print(
    f"SUCCESS: Done! Output now has {len(existing_rows)} rows at {args.output}. "
    f"Processed ids [{effective_from}, {effective_to})."
)
