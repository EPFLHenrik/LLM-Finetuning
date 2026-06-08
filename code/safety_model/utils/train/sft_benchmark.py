from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import tqdm
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from constants.global_variables import DatasetFieldNames, Random
from utils.evaluate.score import main as evaluate_score
from constants.system_prompt import SYSTEM_PROMPT_MCQ

try:
    from vllm import LLM, SamplingParams

    HAS_VLLM = True
except Exception:
    LLM = None
    SamplingParams = None
    HAS_VLLM = False

SEED = Random.SEED
MAX_NEW_TOKENS = 16384
TOP_P = 0.9
TEMPERATURE = 0.7


def run_benchmark_eval(
    checkpoint_path: Path,
    benchmark: str,
    think_mode: bool,
    use_vllm: bool,
    limit: int | None,
    run_name: str,
    epoch: int,
) -> dict:
    checkpoint_path = Path(os.path.abspath(os.path.expanduser(str(checkpoint_path))))
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_path}")

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    dataset = load_from_disk(f"data/processed/benchmark/{benchmark}")
    if limit is not None:
        dataset = dataset.select(range(limit))  # type: ignore

    output_dir = Path(f"code/safety_model/runs/benchmark/outputs/{benchmark}")
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{run_name}_epoch-{epoch}_{'think' if think_mode else 'no_think'}_{limit if limit is not None else 'full'}.jsonl"
    output_path = output_dir / filename

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))

    model = None
    llm = None
    sampling_params = None

    if use_vllm:
        if not HAS_VLLM:
            raise RuntimeError("--benchmark_use_vllm was set but vLLM is not available.")
        llm = LLM(model=str(checkpoint_path), dtype="auto", seed=SEED)
        sampling_params = SamplingParams(
            top_p=TOP_P,
            temperature=TEMPERATURE,
            max_tokens=MAX_NEW_TOKENS,
            seed=SEED,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint_path),
            torch_dtype="auto",
            device_map="auto",
        )

    with output_path.open("w", encoding="utf-8") as f:
        for row in tqdm.tqdm(dataset, desc=f"Benchmark {benchmark}"):  # type: ignore
            row_dict = dict(row)
            prompt = row_dict[DatasetFieldNames.BENCHMARK_INPUT]
            answer = row_dict[DatasetFieldNames.BENCHMARK_ANSWER]

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_MCQ},
                {"role": "user", "content": prompt},
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=think_mode,
            )

            if use_vllm:
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
                DatasetFieldNames.BENCHMARK_INPUT: prompt,
                DatasetFieldNames.BENCHMARK_COMPLETIONS: [completion],
                DatasetFieldNames.BENCHMARK_ANSWER: answer,
            }
            f.write(json.dumps(output_row, ensure_ascii=False) + "\n")

    eval_argv = [
        "--generations",
        str(output_path),
        "--benchmark",
        "safety",
        "--output",
        str(output_path).replace("outputs", "results").replace(".jsonl", "_results.json"),
    ]
    evaluate_score(eval_argv)

    results_path = Path(eval_argv[-1])
    with results_path.open("r", encoding="utf-8") as f:
        results_payload = json.load(f)

    metrics = results_payload.get("metrics", {})
    return {
        "benchmark": benchmark,
        "metrics": {
            "pass@1": metrics.get("pass@1"),
            "box_compliance_rate": metrics.get("box_compliance_rate"),
            "pass@1_among_extracted": metrics.get("pass@1_among_extracted"),
        },
        "output_path": str(output_path),
        "results_path": str(results_path),
    }