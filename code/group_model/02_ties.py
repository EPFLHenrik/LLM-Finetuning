from code.group_model.utils.parse_arguments import parse_merge_args
import subprocess
import yaml
args = parse_merge_args()

DENSITY = 0.5
NORMALIZE = True
INT8_MASK = True


def build_config(args):
    return {
        "merge_method": "ties",
        "base_model": args.base_model,
        "models": [
            {
                "model": model,
                "parameters": {
                    "weight": weight,
                    "density": DENSITY,
                },
            }
            for model, weight in zip(args.models, args.weights)
        ],
        "parameters": {
            "normalize": NORMALIZE,
            "int8_mask": INT8_MASK,
        },
        "dtype": args.dtype,
    }


def main():
    args = parse_merge_args()
    config = build_config(args)

    with open(args.config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    subprocess.run(
        ["mergekit-yaml", args.config_path, args.output_path, "--cuda"],
        check=True,
    )


if __name__ == "__main__":
    main()