"""Eval every LoRA checkpoint in a training output dir on MATH-500 (or any
registered dataset) and report which one is best.

For each `checkpoint-N/` directory:
  1. Merges the LoRA adapter into Qwen3-1.7B in a temp dir
  2. Runs `math_task.run_eval` on the given dataset/budget
  3. Reads the resulting *_scored.json summary
  4. Appends a row to `math_task/RUNS.md`
  5. Deletes the temp merged dir to save scratch space

At the end, prints a sorted table by pass@1 so you can pick the best
checkpoint to push to HuggingFace via push_checkpoint.py.

Usage:
    python -m math_task.training.eval_all_checkpoints \\
        --ckpt-dir /scratch/user/ckpt/math_sft_v1 \\
        --dataset math500_stratified \\
        --budget final

    # Faster sanity loop (25 problems = 5/level):
    python -m math_task.training.eval_all_checkpoints \\
        --ckpt-dir /scratch/user/ckpt/math_sft_v1 \\
        --dataset math500_stratified \\
        --budget final \\
        --limit 5

    # Only eval certain checkpoints (regex match on the checkpoint dir name):
    python -m math_task.training.eval_all_checkpoints \\
        --ckpt-dir /scratch/user/ckpt/math_sft_v1 \\
        --dataset math500_stratified \\
        --budget final \\
        --filter "checkpoint-(150|300|450|594)$"
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List


DEFAULT_BASE_MODEL = "Qwen/Qwen3-1.7B"
CHAT_TEMPLATE_PATH = Path(__file__).parent / "chat_template.jinja"


def list_checkpoints(ckpt_dir: Path, pattern: str = r"^checkpoint-\d+$") -> List[Path]:
    """Return checkpoint-N subdirs sorted by step number ascending."""
    regex = re.compile(pattern)
    subs = [p for p in ckpt_dir.iterdir() if p.is_dir() and regex.match(p.name)]
    subs.sort(key=lambda p: int(p.name.split("-")[1]))
    return subs


def merge_lora_to_tempdir(ckpt: Path, base_model: str, tmp_dir: Path) -> Path:
    """Merge a LoRA checkpoint with the base model into `tmp_dir`."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    template = CHAT_TEMPLATE_PATH.read_text()
    print(f"[merge] base {base_model} + adapter {ckpt.name}")
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    peft_model = PeftModel.from_pretrained(base, str(ckpt))
    merged = peft_model.merge_and_unload()

    tok = AutoTokenizer.from_pretrained(base_model)
    tok.chat_template = template

    gen_cfg = GenerationConfig(
        bos_token_id=151643,
        eos_token_id=[151645, 151643],
        pad_token_id=151643,
        do_sample=True,
        temperature=0.7, top_p=0.95, top_k=20,
    )

    tmp_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(tmp_dir)
    tok.save_pretrained(tmp_dir)
    gen_cfg.save_pretrained(tmp_dir)
    (tmp_dir / "chat_template.jinja").write_text(template)

    # Free the model from python's memory before vLLM tries to take the GPU.
    del merged
    del base
    del peft_model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return tmp_dir


def run_eval(
    model_path: Path,
    dataset: str,
    budget: str,
    thinking: str,
    limit: int | None,
) -> dict:
    """Spawn `python -m math_task.run_eval` as a subprocess so the vLLM
    engine fully tears down between checkpoints (otherwise CUDA memory
    creeps up across iterations)."""
    cmd = [
        sys.executable, "-m", "math_task.run_eval",
        "--dataset", dataset,
        "--budget", budget,
        "--thinking", thinking,
        "--model", str(model_path),
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    print(f"[eval] running: {' '.join(cmd)}")
    rc = subprocess.run(cmd, check=False).returncode
    if rc != 0:
        print(f"[eval] WARN: run_eval exited with rc={rc}")

    # The result file path is deterministic.
    safe_model = str(model_path).replace("/", "_")
    repo_root = Path(__file__).resolve().parents[2]
    stem = f"{dataset}_{safe_model}_none_think-{thinking}_budget-{budget}"
    scored_path = repo_root / "math_task" / "results" / f"{stem}_scored.json"
    if not scored_path.exists():
        print(f"[eval] WARN: scored file not found: {scored_path}")
        return {}
    with open(scored_path) as f:
        scored = json.load(f)
    return scored.get("summary", {})


def run_eval_v2(
    model_path: Path,
    profile: str,
    budget: str,
    thinking: str,
    n: int,
) -> dict:
    """Spawn `python -m math_task.eval_v2` as a subprocess. Same teardown
    rationale as run_eval(). Returns the eval_v2 summary dict, flattened
    with per-subset pass@1 keys so it slots into the same RUNS.md row schema.
    """
    cmd = [
        sys.executable, "-m", "math_task.eval_v2",
        "--model", str(model_path),
        "--profile", profile,
        "--budget", budget,
        "--thinking", thinking,
        "--n", str(n),
    ]
    print(f"[eval-v2] running: {' '.join(cmd)}")
    rc = subprocess.run(cmd, check=False).returncode
    if rc != 0:
        print(f"[eval-v2] WARN: exited with rc={rc}")

    safe_model = str(model_path).replace("/", "_")
    repo_root = Path(__file__).resolve().parents[2]
    out_path = (
        repo_root / "math_task" / "results" / "eval_v2"
        / f"{safe_model}_{budget}_{profile}_think-{thinking}_n-{n}.json"
    )
    if not out_path.exists():
        print(f"[eval-v2] WARN: result file not found: {out_path}")
        return {}
    with open(out_path) as f:
        result = json.load(f)

    summary = result.get("summary", {})
    agg = summary.get("aggregate", {})
    flat = {
        "weighted_pass@1": agg.get("weighted_pass@1"),
        "weighted_pass@8": agg.get("weighted_pass@8"),
        "thinking": thinking,
        "budget": budget,
        "profile": profile,
        "n": n,
    }
    # Hoist per-subset pass@1/pass@8 onto the row so the leaderboard prints them.
    for subset, s in summary.get("per_subset", {}).items():
        metrics = s.get("metrics", {})
        flat[f"{subset}_p@1"] = metrics.get("pass@1")
        if "pass@8" in metrics:
            flat[f"{subset}_p@8"] = metrics.get("pass@8")
    return flat


def append_runs_md(row: dict, runs_md: Path) -> None:
    """Append a row to RUNS.md (creates the file with a header if missing)."""
    header = (
        "| date | checkpoint | dataset | budget | think | pass@1 | pass@8 | box% | trunc% | avg_tok |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    if not runs_md.exists():
        runs_md.write_text("# Math task evaluation runs\n\n" + header)

    line = (
        f"| {row.get('date','?')} "
        f"| {row.get('checkpoint','?')} "
        f"| {row.get('dataset','?')} "
        f"| {row.get('budget','?')} "
        f"| {row.get('thinking','?')} "
        f"| {row.get('pass@1', 0):.4f} "
        f"| {row.get('pass@8', 0):.4f} "
        f"| {row.get('box_compliance_rate', 0):.4f} "
        f"| {row.get('truncation_rate', 0):.4f} "
        f"| {row.get('avg_tokens_per_completion', 0):.1f} |\n"
    )
    with open(runs_md, "a") as f:
        f.write(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path, required=True,
                    help="Training output dir containing checkpoint-N subdirs.")
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                    help=f"HF repo or local path to merge each LoRA into. "
                         f"Default {DEFAULT_BASE_MODEL}. For RL-on-top-of-SFT "
                         f"eval, pass cs-552-2026-mystery-machine/math_model "
                         f"(which already has SFT weights merged in).")
    ap.add_argument("--dataset", default="math500_stratified",
                    help="Dataset key (see run_eval DATASET_LOADERS). "
                         "Ignored when --use-eval-v2 is set.")
    ap.add_argument("--budget", default="final", choices=["ci", "final"])
    ap.add_argument("--thinking", default="on", choices=["on", "off"])
    ap.add_argument("--limit", type=int, default=None,
                    help="Pass-through to run_eval. For math500_stratified, "
                         "this means n_per_level (5 -> 25 problems total). "
                         "Ignored when --use-eval-v2 is set.")
    ap.add_argument("--use-eval-v2", action="store_true",
                    help="Call math_task.eval_v2 instead of math_task.run_eval. "
                         "Use this when ranking checkpoints against the "
                         "calibrated multi-dataset weighted eval.")
    ap.add_argument("--eval-v2-profile", default="quick",
                    help="When --use-eval-v2: which eval_v2 profile to run "
                         "(quick | fast | broad | default | full).")
    ap.add_argument("--eval-v2-n", type=int, default=8,
                    help="When --use-eval-v2: generations per problem. "
                         "Default 8 to match the final-grade pass@8 target.")
    ap.add_argument("--filter", default=r"^checkpoint-\d+$",
                    help="Regex to filter checkpoint subdir names.")
    ap.add_argument("--temp-root", type=Path, default=Path("/tmp/sft_merge"),
                    help="Where to put merged temp dirs (deleted after each eval).")
    ap.add_argument("--runs-md", type=Path, default=None,
                    help="Default: math_task/RUNS.md at repo root.")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    runs_md = args.runs_md or (repo_root / "math_task" / "RUNS.md")

    ckpts = list_checkpoints(args.ckpt_dir, args.filter)
    if not ckpts:
        print(f"No checkpoints matching {args.filter!r} under {args.ckpt_dir}")
        sys.exit(1)
    print(f"[main] {len(ckpts)} checkpoints to evaluate: {[c.name for c in ckpts]}")

    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d %H:%M")

    summaries = []
    for ckpt in ckpts:
        print(f"\n{'=' * 72}\n  {ckpt.name}\n{'=' * 72}")
        tmp_dir = args.temp_root / ckpt.name
        try:
            merge_lora_to_tempdir(ckpt, args.base_model, tmp_dir)
        except Exception as e:
            print(f"[merge] FAILED for {ckpt.name}: {e}")
            continue

        if args.use_eval_v2:
            summary = run_eval_v2(
                tmp_dir, args.eval_v2_profile, args.budget, args.thinking, args.eval_v2_n,
            )
        else:
            summary = run_eval(tmp_dir, args.dataset, args.budget, args.thinking, args.limit)
        summary["checkpoint"] = ckpt.name
        summary["date"] = today
        summaries.append(summary)

        # Append row immediately so we don't lose progress if something later crashes.
        append_runs_md(summary, runs_md)
        if args.use_eval_v2:
            w1 = summary.get("weighted_pass@1")
            w8 = summary.get("weighted_pass@8")
            line = f"[main] {ckpt.name}: weighted_pass@1={w1:.4f}" if w1 is not None else f"[main] {ckpt.name}: (no result)"
            if w8 is not None:
                line += f" weighted_pass@8={w8:.4f}"
            print(line)
        else:
            print(f"[main] {ckpt.name}: pass@1={summary.get('pass@1'):.4f} "
                  f"pass@8={summary.get('pass@8'):.4f}")

        # Free disk: drop the merged temp dir.
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Final leaderboard. When using eval_v2, sort by weighted_pass@8 if
    # available (matches the final-grade target) else weighted_pass@1.
    if args.use_eval_v2:
        key_metric = "weighted_pass@8" if any(s.get("weighted_pass@8") is not None for s in summaries) else "weighted_pass@1"
        print(f"\n{'=' * 72}\n  LEADERBOARD (sorted by {key_metric})\n{'=' * 72}")
        summaries.sort(key=lambda s: s.get(key_metric) or 0, reverse=True)
        # Header: checkpoint + the aggregate + per-subset cells.
        per_sub_keys = sorted({k for s in summaries for k in s if k.endswith("_p@1") or k.endswith("_p@8")})
        print(f"{'checkpoint':>16}  {'wp@1':>7}  {'wp@8':>7}  " +
              "  ".join(f"{k:>14}" for k in per_sub_keys))
        for s in summaries:
            cells = "  ".join(
                f"{(s.get(k) or 0):>14.4f}" if s.get(k) is not None else f"{'—':>14}"
                for k in per_sub_keys
            )
            wp1 = s.get("weighted_pass@1") or 0
            wp8 = s.get("weighted_pass@8")
            print(
                f"{s['checkpoint']:>16}  "
                f"{wp1:7.4f}  "
                f"{(wp8 if wp8 is not None else 0):7.4f}  "
                + cells
            )
    else:
        print(f"\n{'=' * 72}\n  LEADERBOARD (sorted by pass@1)\n{'=' * 72}")
        summaries.sort(key=lambda s: s.get("pass@1", 0), reverse=True)
        print(f"{'checkpoint':>16}  {'pass@1':>7}  {'pass@8':>7}  {'box%':>6}  {'trunc%':>7}  {'avg_tok':>7}")
        for s in summaries:
            print(
                f"{s['checkpoint']:>16}  "
                f"{s.get('pass@1', 0):7.4f}  "
                f"{s.get('pass@8', 0):7.4f}  "
                f"{s.get('box_compliance_rate', 0):6.3f}  "
                f"{s.get('truncation_rate', 0):7.3f}  "
                f"{s.get('avg_tokens_per_completion', 0):7.0f}"
            )
    if summaries:
        best = summaries[0]
        best_key = "weighted_pass@8" if args.use_eval_v2 else "pass@1"
        print(f"\n[best] {best['checkpoint']} with {best_key}={best.get(best_key, 0):.4f}")
        print(f"[best] push it with:")
        print(f"        python -m math_task.training.push_checkpoint "
              f"{args.ckpt_dir / best['checkpoint']}")


if __name__ == "__main__":
    main()
