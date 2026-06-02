from code.group_model.merging.utils.parse_arguments import parse_merge_args
from code.group_model.merging.utils.peft_merge import run_mergekit_merge


MERGE_METHOD = "task_arithmetic"


def main():
    args = parse_merge_args()
    run_mergekit_merge(
        args,
        MERGE_METHOD,
        model_parameters_builder=lambda _model, weight: {"weight": weight},
    )


if __name__ == "__main__":
    main()