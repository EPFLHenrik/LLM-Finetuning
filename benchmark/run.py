from constants.global_variables import DatasetFieldNames
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import json
import os
import torch
import tqdm
from evaluate.score import main as evaluate_score

parser = argparse.ArgumentParser(
    description="Generate short SafetyBench rationales with a teacher model."
)
parser.add_argument("--benchmark", type=str, required=True, help="Name of the benchmark to run")
parser.add_argument("--limit", type=int, default=None, help="Limit the number of examples to run (for testing)")
parser.add_argument("--think", action="store_true", help="Whether to use think mode for the model. (default: False)")
parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B", help="Model to evaluate (default: Qwen/Qwen3-1.7B)")
args = parser.parse_args()

MODEL = args.model
THINK_MODE = args.think
OUTPUT_DIR = f"runs/benchmark/outputs/{args.benchmark}/"
JSONL_FILENAME = f"{MODEL.replace('/', '_')}_{'think' if THINK_MODE else 'no_think'}_{args.limit if args.limit is not None else 'full'}.jsonl"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, JSONL_FILENAME)
MAX_NEW_TOKENS = 16384  # max token defined in the report.


print("BENCHMARK EVALUATION")
print(f"Benchmark: {args.benchmark}")
print(f"Model: {MODEL}")
print(f"Think mode: {THINK_MODE}")
print("=" * 50, "\n")

dataset = load_from_disk(f"data/processed/benchmark/{args.benchmark}")
if args.limit is not None:
    dataset = dataset.select(range(args.limit)) # type: ignore
    print(f"Running on limited dataset with {len(dataset)} examples (limit={args.limit})")
else:
    print(f"Running on full dataset with {len(dataset)} examples")

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
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
            {"role": "system", "content": "You are a helpful assistant. Your response must end by encapsulating the final answer in a \\boxed{...} item.\n For example, if the final answer is A, you must end your response with \\boxed{A}.\n\n"},
            {"role": "user", "content": PROMPT}
            ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=THINK_MODE,
        )

        model_inputs = tokenizer([text], return_tensors="pt")
        model_inputs = model_inputs.to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
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