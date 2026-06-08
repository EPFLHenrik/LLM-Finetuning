from multilinguality_model.merging.utils.parse_arguments import parse_merge_args
from multilinguality_model.merging.utils.peft_merge import run_mergekit_merge

MERGE_METHOD = "slerp"


def main():
    args = parse_merge_args()

    if len(args.merging_targets) != 2:
        raise ValueError("SLERP expects exactly two full-model merge targets.")

    if args.weights is None or len(args.weights) != 2:
        raise ValueError("SLERP expects exactly two weights.")

    # Use first model as base for SLERP
    args.base_model = args.merging_targets[0]

    tuned_weight = args.weights[1]

    run_mergekit_merge(
        args,
        MERGE_METHOD,
        model_parameters_builder=lambda _model, _weight: {},
        parameters={"t": tuned_weight},
    )


if __name__ == "__main__":
    main()