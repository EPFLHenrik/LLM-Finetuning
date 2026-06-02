from code.group_model.merging.utils.parse_arguments import parse_merge_args
from code.group_model.merging.utils.peft_merge import run_mergekit_merge

DENSITY = 0.8
NORMALIZE = True
INT8_MASK = True
MERGE_METHOD = "ties"

def _model_parameters_builder(_model, weight):
    return {
        "weight": weight,
        "density": DENSITY,
    }


def main():
    args = parse_merge_args()
    run_mergekit_merge(
        args,
        MERGE_METHOD,
        model_parameters_builder=_model_parameters_builder,
        parameters={
            "normalize": NORMALIZE,
            "int8_mask": INT8_MASK,
        },
    )


if __name__ == "__main__":
    main()