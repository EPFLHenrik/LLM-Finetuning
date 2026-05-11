from constants.global_variables import DatasetFieldNames, Random
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from datetime import datetime, timezone
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

try:
    import wandb
    HAS_WANDB = True
except Exception:
    wandb = None
    HAS_WANDB = False

SEED = Random.SEED

TABLE_COLUMNS = [
    "timestamp_utc",
    "model_name",
    "benchmark",
    "benchmark_subcategory",
    "think_mode",
    "use_vllm",
    "limit",
    "pass@1",
    "box_compliance_rate",
    "pass@1_among_extracted",
    "results_path",
]


def infer_model_name(checkpoint_path: str, override_name: str | None = None) -> str:
    if override_name:
        return override_name

    for part in Path(checkpoint_path).parts:
        if part.startswith("models--"):
            parts = part.split("--")
            if len(parts) >= 3 and parts[-1]:
                return parts[-1]

    return os.path.basename(checkpoint_path.rstrip("/"))


def split_benchmark_name(benchmark_name: str) -> tuple[str, str]:
    normalized = benchmark_name.strip("/")
    if "/" not in normalized:
        return normalized, ""
    suite, subcategory = normalized.split("/", 1)
    return suite, subcategory


def load_table_rows(table_path: Path) -> list[dict]:
    if not table_path.exists():
        return []

    with open(table_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return payload["rows"]

    if isinstance(payload, list):
        return payload

    raise ValueError(f"Unsupported table format in {table_path}")


def save_table_rows(table_path: Path, rows: list[dict]) -> None:
    table_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "columns": TABLE_COLUMNS,
        "rows": rows,
    }
    with open(table_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

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
    type=bool,
    default=True,
    help="Use vLLM for generation. If not set, uses Transformers generate().",
)
parser.add_argument(
    "--skip_wandb",
    action="store_true",
    help="Skip logging benchmark metrics to Weights & Biases.",
)
parser.add_argument(
    "--wandb_project",
    type=str,
    default="MNLP-Safety-Benchmarks",
    help="W&B project name for benchmark table logging.",
)
parser.add_argument(
    "--wandb_entity",
    type=str,
    default=None,
    help="Optional W&B entity/team name.",
)
parser.add_argument(
    "--wandb_run_name",
    type=str,
    default=None,
    help="Optional W&B run name. Defaults to benchmark-<model>-<benchmark>.",
)
parser.add_argument(
    "--wandb_table_path",
    type=Path,
    default=Path("runs/benchmark/results/wandb_benchmark_table.json"),
    help="Local JSON file that stores cumulative benchmark rows before syncing to W&B.",
)
args = parser.parse_args()

# Set seeds for reproducibility
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

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
TOP_P = 0.9
TEMPERATURE = 0.7


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
    llm = LLM(model=CHECKPOINT_PATH, dtype="auto", seed=SEED)
    sampling_params = SamplingParams(
        top_p=TOP_P,
        temperature=TEMPERATURE,
        max_tokens=MAX_NEW_TOKENS,
        seed=SEED,
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

# Collect all prompts and metadata
print("Preparing prompts for batch generation...")
prompts_list = []
metadata_list = []

for row in tqdm.tqdm(dataset, desc="Preparing prompts"):  # type: ignore
    row_dict = dict(row)
    prompt_text = row_dict[DatasetFieldNames.BENCHMARK_INPUT]
    answer_text = row_dict[DatasetFieldNames.BENCHMARK_ANSWER]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_MCQ},
        {"role": "user", "content": prompt_text}
    ]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=THINK_MODE,
    )
    
    prompts_list.append(formatted_prompt)
    metadata_list.append({
        "input": prompt_text,
        "answer": answer_text,
    })

# Generate all outputs at once
print(f"Generating outputs for {len(prompts_list)} examples...")
if args.use_vllm:
    outputs = llm.generate(prompts_list, sampling_params, use_tqdm=True)  # type: ignore[arg-type]
    completions = [output.outputs[0].text.strip() for output in outputs]
else:
    model_inputs = tokenizer(prompts_list, return_tensors="pt", padding=True)
    model_inputs = model_inputs.to(model.device)  # type: ignore[union-attr]

    with torch.no_grad():
        generated_ids = model.generate(  # type: ignore[union-attr]
            **model_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
        )

    prompt_lengths = model_inputs["input_ids"].shape[-1]
    completion_ids = generated_ids[:, prompt_lengths:]
    completions = tokenizer.batch_decode(
        completion_ids,
        skip_special_tokens=True,
    )
    completions = [c.strip() for c in completions]

# Write all outputs
print("Writing results to file...")
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for metadata, completion in tqdm.tqdm(zip(metadata_list, completions), desc="Writing outputs", total=len(metadata_list)):
        output_row = {
            DatasetFieldNames.BENCHMARK_INPUT: metadata["input"],
            DatasetFieldNames.BENCHMARK_COMPLETIONS: [completion],
            DatasetFieldNames.BENCHMARK_ANSWER: metadata["answer"],
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

results_path = Path(eval_argv[-1])

if not args.skip_wandb:
    if not HAS_WANDB:
        raise RuntimeError("--wandb was set but wandb is not installed/importable in this environment.")

    with open(results_path, "r", encoding="utf-8") as f:
        results_payload = json.load(f)

    metrics = results_payload.get("metrics", {})
    benchmark_suite, benchmark_subcategory = split_benchmark_name(args.benchmark)
    model_name = infer_model_name(CHECKPOINT_PATH, args.name)

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_name": model_name,
        "benchmark": benchmark_suite,
        "benchmark_subcategory": benchmark_subcategory,
        "think_mode": THINK_MODE,
        "use_vllm": args.use_vllm,
        "limit": args.limit,
        "pass@1": metrics.get("pass@1"),
        "box_compliance_rate": metrics.get("box_compliance_rate"),
        "pass@1_among_extracted": metrics.get("pass@1_among_extracted"),
        "results_path": str(results_path),
    }

    table_path = Path(args.wandb_table_path)
    rows = load_table_rows(table_path)
    rows.append(row)
    rows = sorted(
        rows,
        key=lambda r: (
            str(r.get("model_name", "")),
            str(r.get("benchmark", "")),
            str(r.get("benchmark_subcategory", "")),
            str(r.get("timestamp_utc", "")),
        ),
    )
    save_table_rows(table_path, rows)

    run_name = args.wandb_run_name or f"benchmark-{model_name}-{args.benchmark.replace('/', '_')}"
    wb_run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=run_name,
        group=f"benchmark/{model_name}",
        job_type="benchmark_eval",
        config={
            "benchmark": args.benchmark,
            "model_name": model_name,
            "checkpoint": CHECKPOINT_PATH,
            "think_mode": THINK_MODE,
            "use_vllm": args.use_vllm,
            "limit": args.limit,
            "results_path": str(results_path),
        },
    )

    table = wandb.Table(columns=TABLE_COLUMNS)
    for existing_row in rows:
        table.add_data(*[existing_row.get(col) for col in TABLE_COLUMNS])

    wb_run.log({"benchmark/results_table": table})
    wb_run.summary["pass@1"] = row["pass@1"]
    wb_run.summary["box_compliance_rate"] = row["box_compliance_rate"]
    wb_run.summary["pass@1_among_extracted"] = row["pass@1_among_extracted"]
    wb_run.finish()

    print(f"W&B table updated with {len(rows)} rows: {table_path}")