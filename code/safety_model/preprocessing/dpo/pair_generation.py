from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Any, cast

from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer, GenerationConfig

from code.safety_model.constants.global_variables import DatasetFieldNames, Random
from code.safety_model.utils.evaluate.extract_answer import (
    extract_boxed_answer,
    is_equiv,
    normalize_final_answer,
)
from code.safety_model.constants.system_prompt import SYSTEM_PROMPT_MCQ

try:
    from vllm import LLM, SamplingParams  # type: ignore[import-not-found]

    HAS_VLLM = True
except Exception:
    LLM = None
    SamplingParams = None
    HAS_VLLM = False


def _load_dataset_records(dataset_path: Path) -> list[dict]:
    if dataset_path.is_dir():
        loaded_dataset = load_from_disk(str(dataset_path))
        if isinstance(loaded_dataset, Dataset):
            return [dict(row) for row in loaded_dataset]
        if isinstance(loaded_dataset, DatasetDict):
            if len(loaded_dataset) != 1:
                raise ValueError(
                    f"Expected a single-split dataset at {dataset_path}, found splits: {list(loaded_dataset.keys())}"
                )
            split_dataset = next(iter(loaded_dataset.values()))
            return [dict(row) for row in split_dataset]
        raise TypeError(
            f"Unsupported dataset type loaded from {dataset_path}: {type(loaded_dataset)!r}"
        )

    raise FileNotFoundError(
        f"Dataset path does not exist or is unsupported: {dataset_path}"
    )


def _shuffle_records(records: list[dict]) -> list[dict]:
    indices = list(range(len(records)))
    rng = random.Random(Random.SEED)
    rng.shuffle(indices)
    return [records[index] for index in indices]


def _get_expected_answer(row: dict) -> str:
    value = row.get(DatasetFieldNames.DS_ANSWER)
    if value is None:
        raise KeyError("Each row must contain an expected answer under 'answer'.")
    return str(value)


def _answers_match(expected_answer: str, model_answer: str) -> bool:
    extracted_answer = extract_boxed_answer(model_answer, strip_double_curly_brace=True)
    candidate_answer = (
        extracted_answer if extracted_answer is not None else model_answer.strip()
    )
    return is_equiv(
        normalize_final_answer(candidate_answer),
        normalize_final_answer(expected_answer),
    )


def _build_prompts(
    tokenizer: AutoTokenizer,
    rows: list[dict],
    system_prompt: str,
) -> list[str]:
    prompts: list[str] = []
    chat_tokenizer = cast(Any, tokenizer)
    for row in rows:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(row[DatasetFieldNames.DS_INPUT])},
        ]
        prompt_text = chat_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prompts.append(prompt_text)
    return prompts


def _load_sampling_params(checkpoint_path: Path, tokenizer: Any) -> Any:
    generation_config = GenerationConfig.from_pretrained(str(checkpoint_path))
    top_p = getattr(generation_config, "top_p", None)
    temperature = getattr(generation_config, "temperature", None)
    top_k = getattr(generation_config, "top_k", None)
    max_tokens = getattr(generation_config, "max_new_tokens", None)
    if max_tokens is None:
        max_tokens = getattr(tokenizer, "model_max_length", None)
    if max_tokens is None:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} does not define a generation length in generation_config.json or tokenizer.model_max_length."
        )
    if top_p is None or temperature is None or top_k is None:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} is missing top_p, temperature, or top_k in generation_config.json."
        )
    assert SamplingParams is not None
    return SamplingParams(
        top_p=top_p,
        temperature=temperature,
        top_k=top_k,
        max_tokens=int(max_tokens),
        seed=Random.SEED,
    )


def _initialize_generator(checkpoint_path: Path) -> tuple[Any, Any, Any]:
    if not HAS_VLLM:
        raise RuntimeError(
            "vLLM is required to run failures_extraction with --checkpoint."
        )

    tokenizer = cast(Any, AutoTokenizer.from_pretrained(str(checkpoint_path)))
    assert LLM is not None and SamplingParams is not None
    llm = LLM(model=str(checkpoint_path), dtype="auto", seed=Random.SEED)
    sampling_params = _load_sampling_params(checkpoint_path, tokenizer)
    return tokenizer, llm, sampling_params


def _generate_responses(
    tokenizer: Any,
    llm: Any,
    sampling_params: Any,
    rows: list[dict],
    system_prompt: str,
) -> list[str]:
    prompts = _build_prompts(tokenizer, rows, system_prompt=system_prompt)
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    if len(outputs) != len(rows):
        raise RuntimeError(
            f"Expected {len(rows)} generations from vLLM, got {len(outputs)}."
        )
    return [output.outputs[0].text.strip() for output in outputs]


def build_failures_dataset(
    records: list[dict],
    start_index: int,
    end_index: int,
    checkpoint_path: Path,
    max_rounds: int,
    system_prompt: str,
) -> tuple[list[dict], dict[str, int]]:
    sliced_records = records[start_index:end_index]
    if max_rounds <= 0:
        raise ValueError(f"max_rounds must be > 0, got {max_rounds}")

    tokenizer, llm, sampling_params = _initialize_generator(checkpoint_path)

    states: list[dict[str, Any]] = []
    for row in sliced_records:
        states.append(
            {
                "row": row,
                "has_good": False,
                "has_bad": False,
                "chosen": "",
                "rejected": "",
                "rounds": 0,
            }
        )

    active_indices = list(range(len(sliced_records)))
    for _ in range(max_rounds):
        if not active_indices:
            break

        round_rows = [sliced_records[index] for index in active_indices]
        model_generations = _generate_responses(
            tokenizer,
            llm,
            sampling_params,
            round_rows,
            system_prompt=system_prompt,
        )

        next_active_indices: list[int] = []
        for index, model_generation in zip(active_indices, model_generations):
            state = states[index]
            state["rounds"] += 1

            expected_answer = _get_expected_answer(cast(dict, state["row"]))
            if _answers_match(expected_answer, model_generation):
                if not state["has_good"]:
                    state["has_good"] = True
                    state["chosen"] = model_generation
            else:
                if not state["has_bad"]:
                    state["has_bad"] = True
                    state["rejected"] = model_generation

            has_good = bool(state["has_good"])
            has_bad = bool(state["has_bad"])
            rounds = int(state["rounds"])
            if rounds < max_rounds and not (has_good and has_bad):
                next_active_indices.append(index)

        active_indices = next_active_indices

    failures: list[dict] = []
    total_generations = sum(int(state["rounds"]) for state in states)
    failed_with_chosen = 0
    failed_without_chosen = 0
    all_correct_discarded = 0
    stopped_with_both = 0
    stats = {
        "total_records": len(sliced_records),
        "failed_records": 0,
        "failed_with_chosen": 0,
        "failed_without_chosen": 0,
        "all_correct_discarded": 0,
        "stopped_with_both": 0,
        "total_generations": total_generations,
    }

    for state in states:
        has_bad = bool(state["has_bad"])
        has_good = bool(state["has_good"])
        if has_good and has_bad:
            stopped_with_both += 1

        if not has_bad:
            all_correct_discarded += 1
            continue

        failure_row = dict(cast(dict, state["row"]))
        failure_row[DatasetFieldNames.DPO_REJECTED] = str(state["rejected"])
        failure_row[DatasetFieldNames.DPO_CHOSEN] = (
            str(state["chosen"]) if has_good else ""
        )
        failure_row["DPO_generation_model"] = str(checkpoint_path)
        failures.append(failure_row)

        if has_good:
            failed_with_chosen += 1
        else:
            failed_without_chosen += 1

    stats["failed_records"] = len(failures)
    stats["failed_with_chosen"] = failed_with_chosen
    stats["failed_without_chosen"] = failed_without_chosen
    stats["all_correct_discarded"] = all_correct_discarded
    stats["stopped_with_both"] = stopped_with_both

    return failures, stats


def _write_log_file(
    log_path: Path,
    dataset_path: Path,
    output_path: Path,
    start_index: int,
    end_index: int,
    stats: dict[str, int],
) -> None:
    failure_rate = (
        stats["failed_records"] / stats["total_records"]
        if stats["total_records"] > 0
        else 0.0
    )

    lines = [
        f"dataset_path={dataset_path}",
        f"output_path={output_path}",
        f"start_index={start_index}",
        f"end_index={end_index}",
        f"total_records={stats['total_records']}",
        f"failed_records={stats['failed_records']}",
        f"failed_with_chosen={stats['failed_with_chosen']}",
        f"failed_without_chosen={stats['failed_without_chosen']}",
        f"all_correct_discarded={stats['all_correct_discarded']}",
        f"stopped_with_both={stats['stopped_with_both']}",
        f"total_generations={stats['total_generations']}",
        f"failure_rate={failure_rate:.6f}",
    ]

    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="dpo.failures_extraction",
        description="Extract benchmark failures into a DPO dataset.",
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset_path", required=True, type=Path)
    parser.add_argument("--output_path", required=True, type=Path)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--max_rounds", type=int, default=8)
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=SYSTEM_PROMPT_MCQ,
        help="System prompt used when generating chosen/rejected candidates.",
    )
    args = parser.parse_args()

    if not HAS_VLLM:
        raise RuntimeError(
            "vLLM is required for checkpoint-based failure extraction, but it is not available."
        )
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {args.checkpoint}"
        )

    records = _load_dataset_records(args.dataset_path)
    print(
        "WARNING: Shuffling the dataset records before extraction. This ensures that the shuffling is the same as SFT to enable selecting different indices!"
    )
    records = _shuffle_records(records)
    dataset_size = len(records)

    start_index = args.start_index
    end_index = dataset_size if args.end_index is None else args.end_index

    if start_index < 0:
        raise ValueError(f"start_index must be >= 0, got {start_index}")
    if end_index < 0:
        raise ValueError(f"end_index must be >= 0, got {end_index}")
    if start_index > end_index:
        raise ValueError(
            f"start_index must be <= end_index, got start_index={start_index}, end_index={end_index}"
        )
    if end_index > dataset_size:
        raise IndexError(
            f"end_index={end_index} is out of range for dataset size {dataset_size}"
        )

    failures, stats = build_failures_dataset(
        records,
        start_index,
        end_index,
        args.checkpoint,
        args.max_rounds,
        args.system_prompt,
    )

    if args.output_path.exists():
        if args.output_path.is_dir():
            shutil.rmtree(args.output_path)
        else:
            args.output_path.unlink()

    args.output_path.mkdir(parents=True, exist_ok=True)

    failures_dataset = Dataset.from_list(failures)
    failures_dataset.save_to_disk(str(args.output_path))

    log_path = args.output_path / "failures_extraction.log"
    _write_log_file(
        log_path, args.dataset_path, args.output_path, start_index, end_index, stats
    )

    print(
        f"Saved {stats['failed_records']} failures out of {stats['total_records']} examined rows to {args.output_path}"
    )
    print(f"Wrote log to {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())