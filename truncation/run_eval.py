"""Entry point.

Run from the repo root:

    # Smoke test on the 10 frozen validation problems (~1 min)
    python -m math.run_eval --dataset validation

    # Real signal on MATH-500 (~30-60 min on a single A100)
    python -m math.run_eval --dataset math500

    # Faster iteration on a subset
    python -m math.run_eval --dataset math500 --limit 50

    # Once you have a finetune, point at any HF checkpoint
    python -m math.run_eval --dataset math500 --model cs-552-2026-<org>/math_model
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .config import EvalConfig
from .dataset_loader import load_validation_samples, load_gsm8k, load_math500
from .generation import run_generation
from .evaluation import compute_metrics


DATASET_LOADERS = {
    "validation": lambda cfg, limit: load_validation_samples(cfg.repo_root),
    "gsm8k":      lambda cfg, limit: load_gsm8k(limit=limit),
    "math500":    lambda cfg, limit: load_math500(limit=limit),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASET_LOADERS))
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap number of problems (useful for quick iteration)")
    ap.add_argument("--model", default=None,
                    help="Override config.model_id, e.g. a finetuned checkpoint")
    ap.add_argument("--out", default=None,
                    help="Output JSON path (default: math/results/<dataset>_<model>.json)")
    args = ap.parse_args()

    cfg = EvalConfig()
    if args.model:
        cfg.model_id = args.model

    items = DATASET_LOADERS[args.dataset](cfg, args.limit)
    print(f"[load] {len(items)} problems from {args.dataset}")

    results = run_generation(items, cfg)
    metrics = compute_metrics(results)

    print("\n=== metrics ===")
    print(json.dumps(metrics, indent=2))

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    safe_model = cfg.model_id.replace("/", "_")
    out_path = Path(args.out) if args.out else (
        cfg.results_dir / f"{args.dataset}_{safe_model}.json"
    )
    payload = {
        "config": {k: str(v) for k, v in asdict(cfg).items()},
        "metrics": metrics,
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()