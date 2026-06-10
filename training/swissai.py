"""Generate teacher answers using the Swiss AI API (Qwen/Qwen3.5-27B).

Discovers every data/preprocessing/*/train.jsonl (skips benchmarks/) and runs
a two-pass strategy per dataset:
  Pass 1 — run every row once, saved to swissai_1.jsonl in the dataset folder.
  Pass 2 — retry wrong answers, saved to swissai_2.jsonl.
  Merged — pass-1 correct + pass-2 results, saved to swissai_merged.jsonl.

If swissai_1.jsonl already exists for a dataset, pass 1 is skipped.

Usage:
    export CSCS_SERVING_API="sk-rc-..."

    python training/swissai.py
    python training/swissai.py --concurrency 8 --temperature 0.7
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import httpx
from tqdm.asyncio import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluate.benchmarks import extract_benchmark_answer, is_correct_benchmark_answer
from evaluate.extract_answer import extract_boxed_answer_with_fallbacks

API_BASE   = "https://api.swissai.svc.cscs.ch/v1"
MODEL      = "Qwen/Qwen3.5-27B"
MAX_TOKENS = 8192

PREPROCESSING_DIR = REPO / "data" / "preprocessing"
SKIP_DIRS         = {"benchmarks"}

DEFAULT_CONCURRENCY = 25
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P       = 0.95

SYSTEM_MESSAGE = (
    "You are a careful reasoning assistant.\n\n"
    "Reason through the problem when needed.\n\n"
    "Your final answer must appear exactly once inside \\boxed{...}.\n\n"
    "Rules:\n"
    "- For multiple-choice questions, output only the correct capital letter inside \\boxed{...}.\n"
    "- For math questions, output only the final answer inside \\boxed{...}.\n"
    "- Do not include explanations or extra text inside the box.\n"
    "- Never output multiple boxed answers.\n"
    "- Never leave the final answer unboxed.\n\n"
    "Examples:\n"
    "\\boxed{B}\n"
    "\\boxed{42}"
)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return rows


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_completion(completion: str, reference: str) -> tuple[str | None, bool, str, str]:
    reference_text = reference.strip().upper()
    if re.fullmatch(r"[A-Z]", reference_text):
        extracted, strategy = extract_boxed_answer_with_fallbacks(completion)
        if extracted is None:
            return None, False, "no_answer_found", strategy or "none"
        correct = extracted == reference_text
        return extracted, correct, "correct" if correct else "wrong_answer", strategy
    extracted = extract_benchmark_answer(completion, "boxed", reference)
    correct = is_correct_benchmark_answer(extracted, reference, "boxed")
    return extracted, correct, "correct" if correct else "wrong_answer", "boxed"


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

async def _call_api(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    api_key: str,
    prompt: str,
    temperature: float,
    top_p: float,
) -> str:
    payload = {
        "model":      MODEL,
        "messages":   [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user",   "content": prompt},
        ],
        "max_tokens":  MAX_TOKENS,
        "temperature": temperature,
        "top_p":       top_p,
        "stream":      False,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    async with sem:
        try:
            resp = await client.post(
                f"{API_BASE}/chat/completions",
                json=payload, headers=headers, timeout=180.0,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"  [API error] {e}")
            return ""


# ---------------------------------------------------------------------------
# Single pass runner
# ---------------------------------------------------------------------------

async def _run_pass(
    rows: list[dict[str, Any]],
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    api_key: str,
    temperature: float,
    top_p: float,
    out_path: Path,
    label: str,
) -> list[dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    correct_count = 0

    with tqdm(total=len(rows), desc=label, unit="ex", dynamic_ncols=True) as pbar:
        async def _process(i: int, row: dict[str, Any]) -> None:
            nonlocal correct_count
            completion = await _call_api(client, sem, api_key, str(row["prompt"]), temperature, top_p)
            reference  = str(row["answer"])
            extracted, correct, reason, strategy = _score_completion(completion, reference)
            correct_count += int(correct)
            results[i] = {
                "prompt":              str(row["prompt"]),
                "answer":              reference,
                "completion":          completion,
                "extracted_answer":    extracted,
                "correct":             correct,
                "reason":              reason,
                "extraction_strategy": strategy,
                "system_message":      SYSTEM_MESSAGE,
            }
            pbar.update(1)
            pbar.set_postfix(correct=correct_count, acc=f"{correct_count/pbar.n:.1%}", refresh=False)

        await asyncio.gather(*[_process(i, row) for i, row in enumerate(rows)])

    ordered = [results[i] for i in range(len(rows))]
    _write_jsonl(out_path, ordered)
    accuracy = correct_count / len(ordered) if ordered else 0.0
    print(f"  [{label}] accuracy: {correct_count}/{len(ordered)} ({accuracy:.2%}) → {out_path.name}")
    return ordered


# ---------------------------------------------------------------------------
# Per-dataset pipeline
# ---------------------------------------------------------------------------

async def _process_dataset(
    train_path: Path,
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    api_key: str,
    temperature: float,
    top_p: float,
) -> None:
    folder     = train_path.parent
    pass1_path = folder / "swissai_1.jsonl"
    pass2_path = folder / "swissai_2.jsonl"
    merged_path = folder / "swissai_merged.jsonl"

    print(f"\n{'='*60}")
    print(f"Dataset: {folder.name}")
    print(f"{'='*60}")

    if merged_path.exists():
        print(f"  Skipping — {merged_path.name} already exists.")
        return

    # Pass 1
    if pass1_path.exists():
        print(f"Pass 1: loading existing {pass1_path.name}")
        pass1_results = _read_jsonl(pass1_path)
    else:
        rows = _read_jsonl(train_path)
        print(f"Pass 1: {len(rows)} rows")
        pass1_results = await _run_pass(
            rows, client, sem, api_key, temperature, top_p,
            pass1_path, "pass1",
        )

    correct  = [r for r in pass1_results if r["correct"]]
    to_retry = [r for r in pass1_results if not r["correct"]]
    print(f"Pass 1 summary: {len(correct)} correct, {len(to_retry)} wrong")

    # Pass 2
    if to_retry:
        print(f"Pass 2: retrying {len(to_retry)} wrong answers")
        pass2_results = await _run_pass(
            to_retry, client, sem, api_key, temperature, top_p,
            pass2_path, "pass2",
        )
    else:
        print("Pass 2: nothing to retry.")
        pass2_results = []

    # Merge — keep only correct rows
    pass2_correct = [r for r in pass2_results if r["correct"]]
    merged = correct + pass2_correct
    _write_jsonl(merged_path, merged)
    print(f"Merged: {len(merged)} correct rows → {merged_path.name}  (pass1={len(correct)}, pass2={len(pass2_correct)}/{len(pass2_results)})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _find_datasets() -> list[Path]:
    return sorted(
        p for p in PREPROCESSING_DIR.glob("*/train.jsonl")
        if p.parent.name not in SKIP_DIRS
    )


async def main_async(args) -> int:
    api_key = os.environ.get("CSCS_SERVING_API", "")
    if not api_key:
        raise SystemExit("CSCS_SERVING_API not set.")

    if args.input:
        datasets = [args.input.expanduser().resolve()]
        if not datasets[0].exists():
            raise SystemExit(f"Input file not found: {datasets[0]}")
    else:
        datasets = _find_datasets()
    if not datasets:
        raise SystemExit(f"No train.jsonl files found under {PREPROCESSING_DIR}")

    print(f"Model: {MODEL}  |  max_tokens={MAX_TOKENS}  |  concurrency={args.concurrency}")
    print(f"Datasets ({len(datasets)}):")
    for p in datasets:
        print(f"  {p.parent.name}/train.jsonl")

    sem = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient() as client:
        for train_path in datasets:
            await _process_dataset(
                train_path, client, sem, api_key,
                args.temperature, args.top_p,
            )

    print("\nAll datasets done.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Two-pass Swiss AI distillation over all preprocessing datasets")
    parser.add_argument("--input",       type=Path,  default=None, help="Single JSONL file to process (default: auto-discover */train.jsonl)")
    parser.add_argument("--concurrency", type=int,   default=DEFAULT_CONCURRENCY)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p",       type=float, default=DEFAULT_TOP_P)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()