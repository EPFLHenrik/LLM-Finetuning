from constants.global_variables import DatasetFieldNames
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import json
import os
import torch
import tqdm
from evaluate.score import main as evaluate_score
from pathlib import Path
from prompts.system_prompt import SYSTEM_PROMPT_MCQ

try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except Exception:
    LLM = None
    SamplingParams = None
    HAS_VLLM = False

parser = argparse.ArgumentParser(
    description="Evaluate a model on a benchmark by generating outputs and then scoring them by extracting answers and comparing to references."
)
parser.add_argument("--benchmark", type=str, required=True, help="Name of the benchmark to run")
parser.add_argument("--limit", type=int, default=None, help="Limit the number of examples to run (for testing)")
parser.add_argument("--think", action="store_true", help="Whether to use think mode for the model. (default: False)")
parser.add_argument("--name", type=str, default=None, help="Optional name to include in the output filename for easier identification. If not set, defaults to using the checkpoint name.")

DEFAULT_HF_HOME = Path(os.environ.get("HF_HOME", "/scratch/hf_cache"))
DEFAULT_CHECKPOINT = (
    DEFAULT_HF_HOME
    / "hub"
    / "models--Qwen--Qwen3-1.7B"
    / "snapshots"
    / "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
)

parser.add_argument(
    "--checkpoint",
    type=Path,
    default=DEFAULT_CHECKPOINT,
    help="Path to a local model checkpoint/snapshot directory to evaluate. Defaults to the cached Qwen3-1.7B snapshot under HF_HOME.",
)
parser.add_argument(
    "--use_vllm",
    action="store_true",
    help="Use vLLM for generation. If not set, uses Transformers generate().",
)
args = parser.parse_args()

CHECKPOINT_PATH = os.path.abspath(os.path.expanduser(args.checkpoint))
if not os.path.isdir(CHECKPOINT_PATH):
    raise FileNotFoundError(f"Checkpoint path does not exist or is not a directory: {CHECKPOINT_PATH}")

THINK_MODE = args.think
OUTPUT_DIR = f"runs/benchmark/outputs/{args.benchmark}/"
checkpoint_name = os.path.basename(CHECKPOINT_PATH.rstrip("/"))
JSONL_FILENAME = f"{checkpoint_name}_{'think' if THINK_MODE else 'no_think'}_{args.limit if args.limit is not None else 'full'}.jsonl"
JSONL_FILENAME = f"{args.name}_{'think' if THINK_MODE else 'no_think'}_{args.limit if args.limit is not None else 'full'}.jsonl" if args.name else JSONL_FILENAME
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, JSONL_FILENAME)
MAX_NEW_TOKENS = 16384  # max token defined in the report.


print("BENCHMARK EVALUATION")
print(f"Benchmark: {args.benchmark}")
print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"Think mode: {THINK_MODE}")
print(f"Use vLLM: {args.use_vllm}")
print("=" * 50, "\n")

dataset = load_from_disk(f"data/processed/benchmark/{args.benchmark}")
if args.limit is not None:
    dataset = dataset.select(range(args.limit)) # type: ignore
    print(f"Running on limited dataset with {len(dataset)} examples (limit={args.limit})")
else:
    print(f"Running on full dataset with {len(dataset)} examples")

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)

model = None
llm = None
sampling_params = None

if args.use_vllm:
    if not HAS_VLLM:
        raise RuntimeError("--use_vllm was set but vLLM is not installed/importable in this environment.")
    llm = LLM(model=CHECKPOINT_PATH, dtype="auto")
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_NEW_TOKENS,
    )
    runtime_device = "vllm"
else:
    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT_PATH,
        torch_dtype="auto",
        device_map="auto",
    )

    try:
        runtime_device = str(next(model.parameters()).device)
    except StopIteration:
        runtime_device = "unknown"

print(f"Runtime device: {runtime_device}")
print(f"Saving outputs to: {OUTPUT_PATH}\n")

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for i, row in enumerate(tqdm.tqdm(dataset, desc="Generating outputs")):  # type: ignore
        row_dict = dict(row)
        PROMPT = row_dict[DatasetFieldNames.BENCHMARK_INPUT]
        ANSWER = row_dict[DatasetFieldNames.BENCHMARK_ANSWER]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_MCQ},
            {"role": "user", "content": PROMPT}
            ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=THINK_MODE,
        )

        if args.use_vllm:
            outputs = llm.generate([text], sampling_params, use_tqdm=False)  # type: ignore[arg-type]
            completion = outputs[0].outputs[0].text.strip()
        else:
            model_inputs = tokenizer([text], return_tensors="pt")
            model_inputs = model_inputs.to(model.device)  # type: ignore[union-attr]

            with torch.no_grad():
                generated_ids = model.generate(  # type: ignore[union-attr]
                    **model_inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                )

            prompt_length = model_inputs["input_ids"].shape[-1]
            completion_ids = generated_ids[:, prompt_length:]
            completion = tokenizer.batch_decode(
                completion_ids,
                skip_special_tokens=True,
            )[0].strip()

        output_row = {
            DatasetFieldNames.BENCHMARK_INPUT: PROMPT,
            DatasetFieldNames.BENCHMARK_COMPLETIONS: [completion],
            DatasetFieldNames.BENCHMARK_ANSWER: ANSWER,
        }
        f.write(json.dumps(output_row, ensure_ascii=False) + "\n")

print("\nGeneration complete. Evaluating the outputs...")

# Typical terminal command:
# python evaluate/score.py --generations runs/outputs/example_safety_outputs.jsonl --benchmark safety --output runs/benchmark/example_safety_results.json
eval_argv = [
    "--generations", OUTPUT_PATH,
    "--benchmark", "safety",
    "--output", OUTPUT_PATH.replace("outputs", "results").replace(".jsonl", "_results.json"),
]
evaluate_score(eval_argv)