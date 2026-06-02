from code.group_model.merging.utils.parse_arguments import parse_merge_args
from code.group_model.merging.utils.peft_merge import run_peft_lora_merge


DENSITY = 0.5
COMBINATION_TYPE = "dare_ties"
ADAPTER_NAME = "dare_ties_merged"


def main():
	args = parse_merge_args()

	if not args.adapters:
		raise ValueError("--adapters must contain at least one LoRA adapter path")

	print("Merging adapters with DARE-TIES...")
	run_peft_lora_merge(
		base_model_path=args.base_model,
		adapter_paths=args.adapters,
		output_path=args.output_path,
		weights=args.weights,
		dtype=args.dtype,
		combination_type=COMBINATION_TYPE,
		density=DENSITY,
		merged_adapter_name=ADAPTER_NAME,
	)
	print(f"Done. Saved merged model to: {args.output_path}")


if __name__ == "__main__":
	main()