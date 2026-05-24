from constants.global_variables import DatasetFieldNames, Random
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from datetime import datetime, timezone
import json
import os
import torch
import tqdm
from evaluate.score import main as evaluate_score, score_generations
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

SAFETYBENCH_CATEGORY_FIELD = "category"
SAFETYBENCH_NUM_OPTIONS_FIELD = "num_options"
SGXSTEST_CATEGORY_FIELD = "category"


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


def resolve_benchmark_root() -> Path:
    # Prefer the pluralized path requested by CLI users, while preserving compatibility.
    candidates = [
        Path("data/processed/benchmarks"),
        Path("data/processed/benchmark"),
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    searched = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not find benchmark root directory. Searched: {searched}")


def discover_benchmarks_with_arrow(benchmark_root: Path) -> list[str]:
    dataset_dirs = {arrow_path.parent for arrow_path in benchmark_root.rglob("*.arrow")}
    benchmark_names = sorted(
        str(dataset_dir.relative_to(benchmark_root)).replace(os.sep, "/")
        for dataset_dir in dataset_dirs
    )

    print(f"Found {len(benchmark_names)} benchmark folders containing .arrow files under {benchmark_root}:")
    for benchmark_name in benchmark_names:
        print(f"  - {benchmark_name}")

    return benchmark_names


def build_output_path(
    benchmark_name: str,
    checkpoint_path: str,
    think_mode: bool,
    limit: int | None,
    run_name_override: str | None,
) -> Path:
    output_dir = Path("runs") / "benchmark" / "outputs" / benchmark_name
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_name = os.path.basename(checkpoint_path.rstrip("/"))
    jsonl_filename = f"{checkpoint_name}_{'think' if think_mode else 'no_think'}_{limit if limit is not None else 'full'}.jsonl"
    if run_name_override:
        jsonl_filename = f"{run_name_override}_{'think' if think_mode else 'no_think'}_{limit if limit is not None else 'full'}.jsonl"

    return output_dir / jsonl_filename


def load_jsonl_rows(jsonl_path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {jsonl_path}:{line_no}: {exc}") from exc
    return rows


def compute_safetybench_grouped_metrics(generations_path: Path) -> dict:
    items = load_jsonl_rows(generations_path)

    by_category: dict[str, list[dict]] = {}
    by_num_options: dict[str, list[dict]] = {}

    for item in items:
        category_value = str(item.get(SAFETYBENCH_CATEGORY_FIELD, "UNKNOWN"))
        num_options_value = str(item.get(SAFETYBENCH_NUM_OPTIONS_FIELD, "UNKNOWN"))
        by_category.setdefault(category_value, []).append(item)
        by_num_options.setdefault(num_options_value, []).append(item)

    grouped_payload = {
        "by_category": {},
        "by_num_options": {},
    }

    for category, group_items in sorted(by_category.items()):
        result = score_generations(group_items, method="boxed")
        grouped_payload["by_category"][category] = {
            "n_problems": result["n_problems"],
            "n_completions": result["n_completions"],
            "metrics": result["metrics"],
        }

    for num_options, group_items in sorted(by_num_options.items(), key=lambda kv: kv[0]):
        result = score_generations(group_items, method="boxed")
        grouped_payload["by_num_options"][num_options] = {
            "n_problems": result["n_problems"],
            "n_completions": result["n_completions"],
            "metrics": result["metrics"],
        }

    return grouped_payload


def compute_sgxstest_grouped_metrics(generations_path: Path) -> dict:
    items = load_jsonl_rows(generations_path)

    by_category: dict[str, list[dict]] = {}
    for item in items:
        category_value = str(item.get(SGXSTEST_CATEGORY_FIELD, "UNKNOWN"))
        by_category.setdefault(category_value, []).append(item)

    grouped_payload = {
        "by_category": {},
    }

    for category, group_items in sorted(by_category.items()):
        result = score_generations(group_items, method="boxed")
        grouped_payload["by_category"][category] = {
            "n_problems": result["n_problems"],
            "n_completions": result["n_completions"],
            "metrics": result["metrics"],
        }

    return grouped_payload


def run_single_benchmark(
    benchmark_name: str,
    benchmark_root: Path,
    args: argparse.Namespace,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM | None,
    llm: LLM | None,
    sampling_params: SamplingParams | None,
    checkpoint_path: str,
    think_mode: bool,
) -> None:
    dataset_path = benchmark_root / benchmark_name
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Benchmark dataset directory does not exist: {dataset_path}")

    output_path = build_output_path(
        benchmark_name=benchmark_name,
        checkpoint_path=checkpoint_path,
        think_mode=think_mode,
        limit=args.limit,
        run_name_override=args.name,
    )

    print("\n" + "=" * 50)
    print(f"Running benchmark: {benchmark_name}")
    print(f"Dataset path: {dataset_path}")
    print(f"Saving outputs to: {output_path}")
    print("=" * 50)

    dataset = load_from_disk(str(dataset_path))
    if args.limit is not None:
        dataset = dataset.select(range(args.limit))  # type: ignore
        print(f"Running on limited dataset with {len(dataset)} examples (limit={args.limit})")
    else:
        print(f"Running on full dataset with {len(dataset)} examples")

    print("Preparing prompts for batch generation...")
    prompts_list = []
    metadata_list = []

    for row in tqdm.tqdm(dataset, desc="Preparing prompts"):  # type: ignore
        row_dict = dict(row)
        prompt_text = row_dict[DatasetFieldNames.BENCHMARK_INPUT]
        answer_text = row_dict[DatasetFieldNames.BENCHMARK_ANSWER]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_MCQ},
            {"role": "user", "content": prompt_text},
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=think_mode,
        )

        prompts_list.append(formatted_prompt)
        metadata_list.append({
            "input": prompt_text,
            "answer": answer_text,
            SAFETYBENCH_CATEGORY_FIELD: row_dict.get(SAFETYBENCH_CATEGORY_FIELD),
            SAFETYBENCH_NUM_OPTIONS_FIELD: row_dict.get(SAFETYBENCH_NUM_OPTIONS_FIELD),
        })

    print(f"Generating outputs for {len(prompts_list)} examples...")
    if args.use_vllm:
        if llm is None or sampling_params is None:
            raise RuntimeError("vLLM generation was requested but LLM/sampling params are not initialized.")
        outputs = llm.generate(prompts_list, sampling_params, use_tqdm=True)
        completions = [output.outputs[0].text.strip() for output in outputs]
    else:
        if model is None:
            raise RuntimeError("Transformers generation was requested but the model is not initialized.")
        model_inputs = tokenizer(prompts_list, return_tensors="pt", padding=True)
        model_inputs = model_inputs.to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
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

    print("Writing results to file...")
    with open(output_path, "w", encoding="utf-8") as f:
        for metadata, completion in tqdm.tqdm(zip(metadata_list, completions), desc="Writing outputs", total=len(metadata_list)):
            output_row = {
                DatasetFieldNames.BENCHMARK_INPUT: metadata["input"],
                DatasetFieldNames.BENCHMARK_COMPLETIONS: [completion],
                DatasetFieldNames.BENCHMARK_ANSWER: metadata["answer"],
            }
            if metadata.get(SAFETYBENCH_CATEGORY_FIELD) is not None:
                output_row[SAFETYBENCH_CATEGORY_FIELD] = metadata[SAFETYBENCH_CATEGORY_FIELD]
            if metadata.get(SAFETYBENCH_NUM_OPTIONS_FIELD) is not None:
                output_row[SAFETYBENCH_NUM_OPTIONS_FIELD] = metadata[SAFETYBENCH_NUM_OPTIONS_FIELD]
            f.write(json.dumps(output_row, ensure_ascii=False) + "\n")

    print("Generation complete. Evaluating the outputs...")
    eval_argv = [
        "--generations", str(output_path),
        "--benchmark", "safety",
        "--output", str(output_path).replace("outputs", "results").replace(".jsonl", "_results.json"),
    ]
    evaluate_score(eval_argv)

    results_path = Path(eval_argv[-1])
    benchmark_suite, benchmark_subcategory = split_benchmark_name(benchmark_name)

    if benchmark_suite == "safetybench":
        grouped_metrics = compute_safetybench_grouped_metrics(output_path)
        with open(results_path, "r", encoding="utf-8") as f:
            results_payload = json.load(f)
        results_payload["safetybench_grouped_metrics"] = grouped_metrics
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results_payload, f, ensure_ascii=False, indent=2)
        print(f"Added safetybench grouped metrics to: {results_path}")

    if benchmark_suite == "sgxstest":
        grouped_metrics = compute_sgxstest_grouped_metrics(output_path)
        with open(results_path, "r", encoding="utf-8") as f:
            results_payload = json.load(f)
        results_payload["sgxstest_grouped_metrics"] = grouped_metrics
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results_payload, f, ensure_ascii=False, indent=2)
        print(f"Added sgxstest grouped metrics to: {results_path}")

    if not args.skip_wandb:
        if not HAS_WANDB:
            raise RuntimeError("--wandb was set but wandb is not installed/importable in this environment.")

        with open(results_path, "r", encoding="utf-8") as f:
            results_payload = json.load(f)

        metrics = results_payload.get("metrics", {})
        model_name = infer_model_name(checkpoint_path, args.name)

        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model_name": model_name,
            "benchmark": benchmark_suite,
            "benchmark_subcategory": benchmark_subcategory,
            "think_mode": think_mode,
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

        run_name = args.wandb_run_name or f"benchmark-{model_name}-{benchmark_name.replace('/', '_')}"
        wb_run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=run_name,
            group=f"benchmark/{model_name}",
            job_type="benchmark_eval",
            reinit=True,
            config={
                "benchmark": benchmark_name,
                "model_name": model_name,
                "checkpoint": checkpoint_path,
                "think_mode": think_mode,
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


MAX_NEW_TOKENS = 16384  # max token defined in the report.
TOP_P = 0.9
TEMPERATURE = 0.7


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a model on one or more benchmarks by generating outputs and then scoring them."
    )
    parser.add_argument("--benchmark", type=str, nargs="+", default=None, help="One or more benchmark names to run. Ignored when --run_all is set.")
    parser.add_argument("--run_all", action="store_true", help="Run all benchmark dataset folders containing .arrow files under data/processed/benchmarks/ (with fallback to data/processed/benchmark/).")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of examples to run (for testing)")
    parser.set_defaults(think=True)
    parser.add_argument("--no_think", action="store_false", dest="think", help="Disable think mode for the model. (default: think mode enabled)")
    parser.add_argument("--name", type=str, default=None, help="Optional name to include in the output filename for easier identification. If not set, defaults to using the checkpoint name.")

    default_hf_home = Path(os.environ.get("HF_HOME", "/scratch/hf_cache"))
    default_checkpoint = (
        default_hf_home
        / "hub"
        / "models--Qwen--Qwen3-1.7B"
        / "snapshots"
        / "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=default_checkpoint,
        help="Path to a local model checkpoint/snapshot directory to evaluate. Defaults to the cached Qwen3-1.7B snapshot under HF_HOME.",
    )
    parser.add_argument(
        "--use_transformers",
        action="store_true",
        help="Use Transformers generate(). If not set, uses vLLM.",
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
    args.use_vllm = not args.use_transformers

    if not args.run_all and not args.benchmark:
        parser.error("--benchmark is required unless --run_all is set.")

    # Set seeds for reproducibility
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    checkpoint_path = os.path.abspath(os.path.expanduser(args.checkpoint))
    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path does not exist or is not a directory: {checkpoint_path}")

    think_mode = args.think
    benchmark_root = resolve_benchmark_root()

    if args.run_all:
        benchmark_names = discover_benchmarks_with_arrow(benchmark_root)
        if not benchmark_names:
            raise FileNotFoundError(f"No benchmark folders containing .arrow files found under {benchmark_root}")
    else:
        benchmark_names = [b.strip("/") for b in args.benchmark]

    print("BENCHMARK EVALUATION")
    print(f"Benchmark root: {benchmark_root}")
    print(f"Benchmarks to run ({len(benchmark_names)}): {benchmark_names}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Think mode: {think_mode}")
    print(f"Use vLLM: {args.use_vllm}")
    print("=" * 50)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    model = None
    llm = None
    sampling_params = None

    if args.use_vllm:
        if not HAS_VLLM:
            raise RuntimeError("--use_vllm was selected by default, but vLLM is not installed/importable in this environment.")
        llm = LLM(model=checkpoint_path, dtype="auto", seed=SEED)
        sampling_params = SamplingParams(
            top_p=TOP_P,
            temperature=TEMPERATURE,
            max_tokens=MAX_NEW_TOKENS,
            seed=SEED,
        )
        runtime_device = "vllm"
    else:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype="auto",
            device_map="auto",
        )
        try:
            runtime_device = str(next(model.parameters()).device)
        except StopIteration:
            runtime_device = "unknown"

    print(f"Runtime device: {runtime_device}")

    failures: list[tuple[str, str]] = []
    for benchmark_name in benchmark_names:
        try:
            run_single_benchmark(
                benchmark_name=benchmark_name,
                benchmark_root=benchmark_root,
                args=args,
                tokenizer=tokenizer,
                model=model,
                llm=llm,
                sampling_params=sampling_params,
                checkpoint_path=checkpoint_path,
                think_mode=think_mode,
            )
        except Exception as exc:
            if not args.run_all and len(benchmark_names) == 1:
                raise
            failures.append((benchmark_name, str(exc)))
            print(f"[warn] Benchmark failed and will be skipped: {benchmark_name}")
            print(f"[warn] Reason: {exc}")

    if failures:
        print("\n" + "=" * 50)
        print(f"Completed with {len(failures)} benchmark failure(s):")
        for benchmark_name, reason in failures:
            print(f"  - {benchmark_name}: {reason}")
        print("=" * 50)


if __name__ == "__main__":
    main()