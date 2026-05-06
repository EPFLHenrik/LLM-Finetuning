from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import argparse
import json
import torch
from datetime import datetime
from pathlib import Path

from preprocessing.entries import Entries


MODEL_NAME = "Qwen/Qwen3-1.7B"
MAX_NEW_TOKENS = 32768
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


def safe_div(num, den):
	if den == 0:
		return None
	return num / den


def infer_positive_label(true_labels):
	preferred_positive_labels = ["A", "1", "true", "yes", "harmful", "unsafe"]
	lower_to_original = {}
	for label in true_labels:
		if label is None:
			continue
		lower_to_original[str(label).strip().lower()] = str(label)

	for preferred in preferred_positive_labels:
		if preferred.lower() in lower_to_original:
			return lower_to_original[preferred.lower()]

	known_labels = sorted({str(label) for label in true_labels if label is not None})
	if len(known_labels) == 2:
		return known_labels[1]
	if len(known_labels) == 1:
		return known_labels[0]
	return None


def compute_binary_metrics(true_labels, pred_labels):
	labels = sorted({str(x) for x in true_labels if x is not None} | {str(x) for x in pred_labels if x is not None})
	positive_label = infer_positive_label(true_labels)
	negative_label = None
	if positive_label is not None:
		negative_candidates = [label for label in labels if label != positive_label]
		if len(negative_candidates) == 1:
			negative_label = negative_candidates[0]

	tp = fp = tn = fn = 0
	for expected, predicted in zip(true_labels, pred_labels):
		expected_str = None if expected is None else str(expected)
		predicted_str = None if predicted is None else str(predicted)

		if positive_label is not None and expected_str == positive_label and predicted_str == positive_label:
			tp += 1
		elif positive_label is not None and expected_str != positive_label and predicted_str == positive_label:
			fp += 1
		elif positive_label is not None and expected_str == positive_label and predicted_str != positive_label:
			fn += 1
		else:
			tn += 1

	precision = safe_div(tp, tp + fp)
	recall = safe_div(tp, tp + fn)
	f1 = safe_div(2 * tp, 2 * tp + fp + fn)
	specificity = safe_div(tn, tn + fp)
	balanced_accuracy = (
		safe_div((recall + specificity), 2)
		if recall is not None and specificity is not None
		else None
	)

	mcc_den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
	mcc = safe_div((tp * tn) - (fp * fn), mcc_den)

	true_distribution = {}
	pred_distribution = {}
	for label in labels:
		true_distribution[label] = sum(1 for x in true_labels if x is not None and str(x) == label)
		pred_distribution[label] = sum(1 for x in pred_labels if x is not None and str(x) == label)

	unknown_prediction_count = sum(1 for x in pred_labels if x is None)

	return {
		"labels": labels,
		"positive_label": positive_label,
		"negative_label": negative_label,
		"confusion_matrix": {
			"tp": tp,
			"fp": fp,
			"tn": tn,
			"fn": fn,
		},
		"precision": precision,
		"recall": recall,
		"f1": f1,
		"specificity": specificity,
		"balanced_accuracy": balanced_accuracy,
		"mcc": mcc,
		"true_label_distribution": true_distribution,
		"predicted_label_distribution": pred_distribution,
		"unknown_prediction_count": unknown_prediction_count,
	}


def run_benchmark(benchmark, n_examples_arg, enable_thinking, show_output):
	ensure_cuda_available()

	dataset_path = Path("data") / "processed" / "benchmark" / benchmark
	if not dataset_path.exists():
		raise FileNotFoundError(f"Benchmark dataset folder not found: {dataset_path}")

	dataset = load_from_disk(str(dataset_path))
	total_examples = len(dataset)
	n_examples = total_examples if n_examples_arg is None else min(n_examples_arg, total_examples)

	run_timestamp = datetime.now().isoformat(timespec="seconds")
	run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
	output_dir = Path("runs") / "benchmark" / "binary_classification" / f"{benchmark}"
	output_dir.mkdir(parents=True, exist_ok=True)

	model_slug = model_name_to_slug(MODEL_NAME)
	summary_path = output_dir / f"{benchmark}-summary-{model_slug}-{run_id}.json"
	records_path = output_dir / f"{benchmark}-records-{model_slug}-{run_id}.jsonl"

	print(f"Loaded {total_examples} examples from dataset: {dataset_path}")
	print(f"Benchmarking first {n_examples} examples.\n")

	tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
	model = AutoModelForCausalLM.from_pretrained(
		MODEL_NAME,
		torch_dtype="auto",
		device_map="cuda",
	)
	device_name = torch.cuda.get_device_name(0)
	print(f"Runtime device: {model.device} ({device_name})")

	correct_count = 0
	extracted_count = 0
	failed_indices = []
	failed_box_compliance_indices = []
	records = []
	true_labels = []
	pred_labels = []

	for i in range(n_examples):
		example = dataset[i][Entries.BINARY_CLASSIFICATION_PROMPT]
		expected_boxed_answer = dataset[i][Entries.BINARY_EXPECTED_BOXED_ANSWER]

		messages = [{"role": "user", "content": example}]
		text = tokenizer.apply_chat_template(
			messages,
			tokenize=False,
			add_generation_prompt=True,
			enable_thinking=enable_thinking,
		)
		model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

		generated_ids = model.generate(
			**model_inputs,
			max_new_tokens=MAX_NEW_TOKENS,
		)

		thinking_content, content = extract_content(tokenizer, model_inputs, generated_ids)
		final_answer = extract_boxed_answer(content)

		if show_output:
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

		true_labels.append(expected_boxed_answer)
		pred_labels.append(final_answer)

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
	binary_metrics = compute_binary_metrics(true_labels, pred_labels)

	print("=" * 50)
	print("Benchmark summary")
	print(f"Evaluated examples: {n_examples}")
	print(f"Correct predictions: {correct_count}")
	print(f"Accuracy: {accuracy:.2f}%")
	print(f"Boxed answer extraction rate: {extraction_rate:.2f}%")
	print(f"Positive label: {binary_metrics['positive_label']}")
	print(f"Negative label: {binary_metrics['negative_label']}")
	print(
		"Confusion matrix (positive class): "
		f"TP={binary_metrics['confusion_matrix']['tp']}, "
		f"FP={binary_metrics['confusion_matrix']['fp']}, "
		f"TN={binary_metrics['confusion_matrix']['tn']}, "
		f"FN={binary_metrics['confusion_matrix']['fn']}"
	)

	def format_metric(metric_name):
		value = binary_metrics[metric_name]
		if value is None:
			return "N/A"
		return f"{value * 100:.2f}%"

	print(f"Precision: {format_metric('precision')}")
	print(f"Recall (Sensitivity): {format_metric('recall')}")
	print(f"F1 score: {format_metric('f1')}")
	print(f"Specificity: {format_metric('specificity')}")
	print(f"Balanced accuracy: {format_metric('balanced_accuracy')}")
	mcc_value = binary_metrics["mcc"]
	print(f"Matthews correlation coefficient (MCC): {'N/A' if mcc_value is None else f'{mcc_value:.4f}'}")
	print(f"True label distribution: {binary_metrics['true_label_distribution']}")
	print(f"Predicted label distribution: {binary_metrics['predicted_label_distribution']}")
	print(f"Unknown/invalid predicted labels: {binary_metrics['unknown_prediction_count']}")

	one_example = records[0] if records else None
	summary = {
		"run_id": run_id,
		"run_timestamp": run_timestamp,
		"model_name": MODEL_NAME,
		"dataset_path": str(dataset_path),
		"dataset_total_examples": total_examples,
		"evaluated_examples": n_examples,
		"max_new_tokens": MAX_NEW_TOKENS,
		"enable_thinking": enable_thinking,
		"correct_predictions": correct_count,
		"accuracy": accuracy,
		"boxed_answer_extraction_rate": extraction_rate,
		"binary_metrics": binary_metrics,
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


def parse_args():
	parser = argparse.ArgumentParser(description="Run a 2-class benchmark.")
	parser.add_argument(
		"--benchmark",
		type=str,
		required=True,
		help="Benchmark folder name under data/processed/benchmark (e.g. advbench, jbb-behaviors).",
	)
	parser.add_argument(
		"--N",
		type=int,
		default=None,
		help="Number of examples to benchmark. If omitted, all examples are evaluated.",
	)
	parser.add_argument(
		"--think",
		action="store_true",
		help="Enable model thinking mode.",
	)
	parser.add_argument(
		"--show_output",
		action="store_true",
		help="Print model outputs during evaluation.",
	)
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()

	if args.N is not None and args.N <= 0:
		raise ValueError("--N must be a positive integer when provided.")

	run_benchmark(
		benchmark=args.benchmark,
		n_examples_arg=args.N,
		enable_thinking=args.think,
		show_output=args.show_output,
	)
