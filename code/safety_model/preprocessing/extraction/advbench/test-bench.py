from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import argparse
import json
import torch
from datetime import datetime
from pathlib import Path

from preprocessing.entries import Entries

model_name = "Qwen/Qwen3-1.7B"
N = 520
MAX_NEW_TOKENS = 32768
ENABLE_THINKING = False
SHOW_MODEL_OUTPUT = False
BOXED_ANSWER_REGEX = r"\\boxed\{([^}]*)\}"


def ensure_cuda_available():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this Python environment. "
            "Install a CUDA-enabled PyTorch build and retry. "
            f"Detected torch version: {torch.__version__}."
        )


def model_name_to_slug(name):
    return name.replace("/", "-")


def save_json(path, payload):
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_content(tokenizer, model_inputs, generated_ids):
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    return thinking_content, content


def extract_boxed_answer(content):
    matches = re.findall(BOXED_ANSWER_REGEX, content)
    if matches:
        if len(matches) > 1:
            print("WARNING: Multiple boxed answers found! Extracting the last one as the final answer.")
        return matches[-1]
    return None


def main(n_examples, output_dir):
    ensure_cuda_available()

    dataset_path = "data/processed/benchmark/advbench"
    dataset = load_from_disk(dataset_path)
    total_examples = len(dataset)
    n_examples = min(n_examples, total_examples)

    run_timestamp = datetime.now().isoformat(timespec="seconds")
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_slug = model_name_to_slug(model_name)
    summary_path = output_dir / f"advbench-summary-{model_slug}-{run_id}.json"
    records_path = output_dir / f"advbench-records-{model_slug}-{run_id}.jsonl"

    print(f"Loaded {total_examples} examples from dataset.")
    print(f"Benchmarking first {n_examples} examples.\n")

    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="cuda"
    )
    device_name = torch.cuda.get_device_name(0)
    print(f"Runtime device: {model.device} ({device_name})")

    correct_count = 0
    extracted_count = 0
    failed_indices = []
    failed_box_compliance_indices = []
    records = []

    for i in range(n_examples):
        example = dataset[i][Entries.BINARY_CLASSIFICATION_PROMPT]
        expected_boxed_answer = dataset[i][Entries.BINARY_EXPECTED_BOXED_ANSWER]

        messages = [{"role": "user", "content": example}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=ENABLE_THINKING
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=MAX_NEW_TOKENS
        )

        thinking_content, content = extract_content(tokenizer, model_inputs, generated_ids)
        final_answer = extract_boxed_answer(content)

        if SHOW_MODEL_OUTPUT:
            print(f"Example {i + 1}: thinking content: {thinking_content}")
            print(f"Example {i + 1}: content: {content}\n")

        if final_answer is not None:
            extracted_count += 1
        else:
            failed_box_compliance_indices.append(i)

        is_correct = final_answer == expected_boxed_answer
        if is_correct:
            correct_count += 1
        else:
            failed_indices.append(i)

        records.append(
            {
                "index": i,
                "prompt": example,
                "expected_boxed_answer": expected_boxed_answer,
                "predicted_boxed_answer": final_answer,
                "has_boxed_answer": final_answer is not None,
                "is_correct": is_correct,
                "response": content,
                "thinking_content": thinking_content,
            }
        )

        print(f"Example {i + 1}/{n_examples}")
        print(f"Expected: {expected_boxed_answer}")
        print(f"Predicted: {final_answer}")
        print(f"Result: {'SUCCESS' if is_correct else 'FAILURE'}\n")

    accuracy = (correct_count / n_examples) * 100 if n_examples > 0 else 0.0
    extraction_rate = (extracted_count / n_examples) * 100 if n_examples > 0 else 0.0

    print("=" * 50)
    print("Benchmark summary")
    print(f"Evaluated examples: {n_examples}")
    print(f"Correct predictions: {correct_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Boxed answer extraction rate: {extraction_rate:.2f}%")

    one_example = records[0] if records else None
    summary = {
        "run_id": run_id,
        "run_timestamp": run_timestamp,
        "model_name": model_name,
        "dataset_path": dataset_path,
        "dataset_total_examples": total_examples,
        "evaluated_examples": n_examples,
        "max_new_tokens": MAX_NEW_TOKENS,
        "enable_thinking": ENABLE_THINKING,
        "correct_predictions": correct_count,
        "accuracy": accuracy,
        "boxed_answer_extraction_rate": extraction_rate,
        "failed_indices": failed_indices,
        "failed_box_compliance_indices": failed_box_compliance_indices,
        "summary_example": {
            "index": one_example["index"],
            "prompt": one_example["prompt"],
            "expected_boxed_answer": one_example["expected_boxed_answer"],
            "predicted_boxed_answer": one_example["predicted_boxed_answer"],
            "is_correct": one_example["is_correct"],
            "response": one_example["response"],
        }
        if one_example
        else None,
        "records_file": str(records_path),
    }

    save_json(summary_path, summary)
    save_jsonl(records_path, records)

    print(f"Summary written to: {summary_path}")
    print(f"Detailed records written to: {records_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AdvBench model benchmark.")
    parser.add_argument(
        "--n",
        type=int,
        default=N,
        help="Number of examples to benchmark."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/advbench_benchmark",
        help="Directory where benchmark artifacts are written."
    )
    args = parser.parse_args()

    if args.n <= 0:
        raise ValueError("--n must be a positive integer.")

    main(args.n, args.output_dir)