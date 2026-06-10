from __future__ import annotations

from pathlib import Path

from datasets import Dataset, Features, Value, concatenate_datasets, load_dataset, load_from_disk
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "data" / "group_model" / "benchmark"

PROMPT_FIELD_CANDIDATES = ("prompt", "question", "input")
ANSWER_FIELD_CANDIDATES = ("answer", "target", "label")


def resolve_column_name(column_names: list[str], candidates: tuple[str, ...], kind: str, source_name: str) -> str:
	for candidate in candidates:
		if candidate in column_names:
			return candidate
	raise ValueError(f"Could not find a {kind} column in {source_name}. Available columns: {column_names}")


def empty_prompt_answer_dataset() -> Dataset:
	return Dataset.from_dict(
		{"prompt": [], "answer": []},
		features=Features({"prompt": Value("string"), "answer": Value("string")}),
	)


def normalize_prompt_answer(dataset: Dataset, source_name: str) -> Dataset:
	prompt_column = resolve_column_name(dataset.column_names, PROMPT_FIELD_CANDIDATES, "prompt", source_name)
	answer_column = resolve_column_name(dataset.column_names, ANSWER_FIELD_CANDIDATES, "answer", source_name)
	rows = [
		{
			"prompt": str(row[prompt_column]),
			"answer": str(row[answer_column]),
		}
		for row in dataset
	]
	return Dataset.from_list(rows)


def save_benchmark(dataset: Dataset, output_dir: Path, source_name: str) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / "benchmark.arrow"
	dataset.save_to_disk(str(output_path))
	saved_dataset = load_from_disk(str(output_path))
	preview = saved_dataset[0] if len(saved_dataset) else {"prompt": None, "answer": None}
	print(f"Saved {source_name} -> {output_path} ({len(saved_dataset)} rows)")
	print(f"Columns: {saved_dataset.column_names}")
	print(f"Preview: {preview}")


def list_repo_paths(api: HfApi, repo_id: str, path_in_repo: str, suffix: str | None = None) -> list[str]:
	items = api.list_repo_tree(repo_id, repo_type="dataset", path_in_repo=path_in_repo, recursive=True)
	paths = [item.path for item in items]
	if suffix is not None:
		paths = [path for path in paths if path.endswith(suffix)]
	return sorted(paths)


def resolve_file_url(repo_id: str, repo_path: str) -> str:
	return f"https://huggingface.co/datasets/{repo_id}/resolve/main/{repo_path}"


def load_json_dataset(url: str) -> Dataset:
	try:
		return load_dataset("json", data_files={"test": url}, split="test")
	except StopIteration:
		return empty_prompt_answer_dataset()


def load_parquet_dataset(url: str) -> Dataset:
	return load_dataset("parquet", data_files={"train": url}, split="train")


def download_safety_benchmarks(api: HfApi) -> None:
	repo_id = "cs-552-2026-mystery-machine/safety"
	parquet_paths = list_repo_paths(api, repo_id, "benchmark", suffix=".parquet")
	for parquet_path in parquet_paths:
		dataset = load_parquet_dataset(resolve_file_url(repo_id, parquet_path))
		normalized = normalize_prompt_answer(dataset, parquet_path)
		relative_output_dir = Path(parquet_path).parent.relative_to("benchmark")
		save_benchmark(normalized, OUTPUT_ROOT / "safety" / relative_output_dir, parquet_path)


def download_multilingual_benchmark() -> None:
	repo_id = "cs-552-2026-mystery-machine/multilingual"
	repo_path = "benchmarks/benchmark_all.jsonl"
	dataset = load_json_dataset(resolve_file_url(repo_id, repo_path))
	normalized = normalize_prompt_answer(dataset, repo_path)
	save_benchmark(normalized, OUTPUT_ROOT / "multilinguality" / "benchmark_all", repo_path)


def download_math_benchmark() -> None:
	dataset = load_dataset("cs-552-2026-mystery-machine/math", split="test")
	normalized = normalize_prompt_answer(dataset, "cs-552-2026-mystery-machine/math:test")
	save_benchmark(normalized, OUTPUT_ROOT / "math" / "math", "math")


def download_general_knowledge_benchmarks(api: HfApi) -> None:
	repo_id = "cs-552-2026-mystery-machine/general_knowledge"
	benchmark_root = "benchmarks"
	top_level_entries = api.list_repo_tree(repo_id, repo_type="dataset", path_in_repo=benchmark_root, recursive=False)
	benchmark_dirs = sorted(item.path for item in top_level_entries if not item.path.endswith(".jsonl"))
	for benchmark_dir in benchmark_dirs:
		test_paths = list_repo_paths(api, repo_id, benchmark_dir, suffix="test.jsonl")
		if not test_paths:
			print(f"Skipping {benchmark_dir}: no test.jsonl files found")
			continue
		available_paths = set(list_repo_paths(api, repo_id, benchmark_dir))
		datasets: list[Dataset] = []
		for test_path in test_paths:
			dataset = load_json_dataset(resolve_file_url(repo_id, test_path))
			if len(dataset) == 0:
				validation_path = test_path.removesuffix("test.jsonl") + "validation.jsonl"
				if validation_path in available_paths:
					print(f"{test_path} is empty; using {validation_path} instead")
					dataset = load_json_dataset(resolve_file_url(repo_id, validation_path))
				else:
					print(f"Skipping empty benchmark file without fallback: {test_path}")
					continue
			datasets.append(normalize_prompt_answer(dataset, test_path))
		if not datasets:
			print(f"Skipping {benchmark_dir}: no non-empty benchmark files found")
			continue
		merged = datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)
		benchmark_name = Path(benchmark_dir).name
		save_benchmark(merged, OUTPUT_ROOT / "general_knowledge" / benchmark_name, benchmark_dir)


def main() -> None:
	api = HfApi()
	download_safety_benchmarks(api)
	download_multilingual_benchmark()
	download_math_benchmark()
	download_general_knowledge_benchmarks(api)

if __name__ == "__main__":
	main()


