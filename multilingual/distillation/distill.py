"""multilingual/distillation/distill.py

Two-stage teacher distillation pipeline.

Stage 1: Both teachers answer independently (answer-only).
         Keep examples where at least one teacher matches gold.
Stage 2: Qwen3.5-27B generates a short same-language rationale
         conditioned on the gold answer.
         Apertus-70B used as tie-breaker when teachers disagree.

Notes:
  - Qwen3.5-27B is a thinking model — responses include <think>...</think>
    blocks. Extraction handles this by reading after </think>.
  - Resumable: appends to existing output files, skips done source_ids.

Output (per dataset/language):
    <output_dir>/<dataset>_<language>_stage1.jsonl
    <output_dir>/<dataset>_<language>_stage2.jsonl

Usage:
    export CSCS_SERVING_API="sk-rc-..."

    # Dry run (5 examples)
    python -m multilingual.distillation.distill \
        --datasets mera --sft_dir multilingual/data/sft/v7 \
        --output_dir multilingual/data/distilled/v7 --dry_run

    # Full run
    python -m multilingual.distillation.distill \
        --datasets mera ceval milu global_mmlu \
        --sft_dir multilingual/data/sft/v7 \
        --output_dir multilingual/data/distilled/v7
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# ---------------------------------------------------------------------------
# API config
# ---------------------------------------------------------------------------

API_BASE = "https://api.swissai.svc.cscs.ch/v1"

# Qwen3.5-27B: thinking model — needs higher max_tokens and longer timeout
TEACHER_PRIMARY   = "Qwen/Qwen3.5-27B"
# Llama: non-thinking, fast
TEACHER_VERIFIER  = "meta-llama/Llama-3.3-70B-Instruct"
# Apertus-70B: thinking model — used as tie-breaker
TEACHER_TIEBREAK  = "swiss-ai/Apertus-70B-Instruct-2509"

# Max tokens per model type
MAX_TOKENS_THINKING    = 800   # for thinking models (Qwen, Apertus)
MAX_TOKENS_NONTHINKING = 20    # for Llama (non-thinking, answer only)
MAX_TOKENS_RATIONALE   = 1200  # for stage 2 rationale generation

DEFAULT_CONCURRENCY = 4    # conservative — thinking models are slower
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 3.0

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

STAGE1_SYSTEM = (
    "You are answering a multiple-choice question. "
    "Return only the option letter: A, B, C, or D. "
    "Do not explain anything."
)

STAGE2_SYSTEM = (
    "You are a multilingual teaching assistant. "
    "You will be given a multiple-choice question and the correct answer. "
    "Write a short explanation (2-4 sentences) in the SAME LANGUAGE as the question "
    "explaining why the given answer is correct. "
    "Be concise and factual. Do not repeat the question. "
    "End your response with the boxed answer on a new line: \\boxed{X} "
    "where X is the correct letter."
)


def _stage1_user(prompt: str) -> str:
    return prompt


def _stage2_user(prompt: str, gold: str) -> str:
    return (
        f"{prompt}\n\n"
        f"The correct answer is {gold}. "
        f"Briefly explain why in the same language as the question, "
        f"then write \\boxed{{{gold}}} on the last line."
    )

# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------

async def _call_api(
    client: httpx.AsyncClient,
    model: str,
    messages: List[Dict],
    api_key: str,
    max_tokens: int = MAX_TOKENS_THINKING,
    temperature: float = 0.0,
    retries: int = DEFAULT_MAX_RETRIES,
) -> Optional[str]:
    """Single API call with retries. Returns raw content string or None."""
    payload = {
        "model":       model,
        "messages":    messages,
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "stream":      False,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    for attempt in range(retries):
        try:
            resp = await client.post(
                f"{API_BASE}/chat/completions",
                json=payload,
                headers=headers,
                timeout=120.0,  # thinking models need more time
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < retries - 1:
                await asyncio.sleep(DEFAULT_RETRY_DELAY * (attempt + 1))
            else:
                print(f"  [API error] {model}: {e}")
                return None
    return None

# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def _strip_think(text: str) -> str:
    """Remove <think>...</think> block from model output."""
    if '</think>' in text:
        return text.split('</think>')[-1].strip()
    # Some models use "Thinking Process:\n...\n</think>" format
    if re.search(r'</think>', text, re.IGNORECASE):
        text = re.split(r'</think>', text, flags=re.IGNORECASE)[-1].strip()
    return text


def _extract_letter(text: Optional[str]) -> Optional[str]:
    """Extract a single A/B/C/D from model output, handling think blocks."""
    if not text:
        return None

    # Strip thinking block first
    cleaned = _strip_think(text).upper()

    # Direct match
    if cleaned in ("A", "B", "C", "D"):
        return cleaned

    # Last non-empty line (most reliable for thinking models)
    lines = [l.strip() for l in cleaned.splitlines() if l.strip()]
    if lines:
        last = lines[-1]
        if last in ("A", "B", "C", "D"):
            return last
        # Last line might be "Answer: B" or "B." etc.
        m = re.search(r'\b([A-D])\b', last)
        if m:
            return m.group(1)

    # Fallback: first standalone letter in cleaned text
    m = re.search(r'\b([A-D])\b', cleaned)
    if m:
        return m.group(1)

    return None


def _extract_boxed(text: Optional[str]) -> Optional[str]:
    """Extract content from \\boxed{X}, handling think blocks."""
    if not text:
        return None
    # Look in the part after </think> first
    cleaned = _strip_think(text)
    m = re.search(r'\\boxed\{([^}]+)\}', cleaned)
    if m:
        return m.group(1).strip().upper()
    # Fallback: search full text
    m = re.search(r'\\boxed\{([^}]+)\}', text)
    if m:
        return m.group(1).strip().upper()
    return None

# ---------------------------------------------------------------------------
# Stage 1: independent answer verification
# ---------------------------------------------------------------------------

async def _stage1_example(
    ex: Dict,
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    api_key: str,
) -> Dict:
    gold   = str(ex["answer"]).strip().upper()
    prompt = ex["messages"][1]["content"]  # user turn only

    messages_s1 = [
        {"role": "system", "content": STAGE1_SYSTEM},
        {"role": "user",   "content": _stage1_user(prompt)},
    ]

    async with sem:
        primary_raw, verifier_raw = await asyncio.gather(
            _call_api(client, TEACHER_PRIMARY,  messages_s1, api_key,
                      max_tokens=MAX_TOKENS_THINKING),
            _call_api(client, TEACHER_VERIFIER, messages_s1, api_key,
                      max_tokens=MAX_TOKENS_NONTHINKING),
        )

    primary_ans  = _extract_letter(primary_raw)
    verifier_ans = _extract_letter(verifier_raw)

    primary_correct  = (primary_ans  == gold)
    verifier_correct = (verifier_ans == gold)

    # Tie-breaker when they disagree
    tiebreak_ans = None
    if primary_correct != verifier_correct:
        async with sem:
            tb_raw = await _call_api(
                client, TEACHER_TIEBREAK, messages_s1, api_key,
                max_tokens=MAX_TOKENS_THINKING,
            )
        tiebreak_ans = _extract_letter(tb_raw)

    any_correct = primary_correct or verifier_correct or (tiebreak_ans == gold)
    confidence  = ("high"   if (primary_correct and verifier_correct) else
                   "medium" if any_correct else
                   "low")

    return {
        **ex,
        "distill": {
            "gold":             gold,
            "primary_raw":      primary_raw,
            "verifier_raw":     verifier_raw,
            "primary_ans":      primary_ans,
            "verifier_ans":     verifier_ans,
            "tiebreak_ans":     tiebreak_ans,
            "primary_correct":  primary_correct,
            "verifier_correct": verifier_correct,
            "any_correct":      any_correct,
            "confidence":       confidence,
        },
    }

# ---------------------------------------------------------------------------
# Stage 2: rationale generation
# ---------------------------------------------------------------------------

async def _stage2_example(
    ex: Dict,
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    api_key: str,
) -> Dict:
    gold   = ex["distill"]["gold"]
    prompt = ex["messages"][1]["content"]

    messages_s2 = [
        {"role": "system", "content": STAGE2_SYSTEM},
        {"role": "user",   "content": _stage2_user(prompt, gold)},
    ]

    async with sem:
        rationale_raw = await _call_api(
            client, TEACHER_PRIMARY, messages_s2, api_key,
            max_tokens=MAX_TOKENS_RATIONALE, temperature=0.3,
        )

    # Extract the clean rationale (after think block)
    rationale_clean = _strip_think(rationale_raw) if rationale_raw else None

    # Validate
    boxed = _extract_boxed(rationale_raw)
    valid = False
    if rationale_clean and boxed == gold:
        n_words = len(rationale_clean.split())
        valid = 5 < n_words < 300

    result = {
        **ex,
        "distill": {
            **ex["distill"],
            "rationale_raw":   rationale_raw,
            "rationale_clean": rationale_clean,
            "rationale_valid": valid,
            "rationale_boxed": boxed,
        },
    }

    # Replace assistant message with clean rationale if valid
    if valid and rationale_clean:
        result["messages"] = list(ex["messages"])
        result["messages"][2] = {
            "role":    "assistant",
            "content": rationale_clean,
        }

    return result

# ---------------------------------------------------------------------------
# File-level pipeline
# ---------------------------------------------------------------------------

async def _distill_file(
    src_path: Path,
    stage1_path: Path,
    stage2_path: Path,
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    api_key: str,
    dry_run: bool,
) -> Dict[str, Any]:
    with open(src_path, encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]

    # Only train split, only MCQ
    records = [r for r in records
               if r.get("split") == "train" and r.get("task_type") == "mcq"]

    if dry_run:
        records = records[:5]

    if not records:
        return {"total": 0, "stage1_pass": 0, "stage2_valid": 0}

    # Resume stage 1
    done_ids: set = set()
    stage1_results: List[Dict] = []
    if stage1_path.exists():
        with open(stage1_path, encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    r = json.loads(l)
                    stage1_results.append(r)
                    done_ids.add(r["source_id"])
        print(f"  Resuming stage1: {len(done_ids)} already done")

    todo = [r for r in records if r["source_id"] not in done_ids]
    print(f"  Stage 1: {len(todo)} remaining / {len(records)} total")

    if todo:
        tasks = [_stage1_example(ex, client, sem, api_key) for ex in todo]
        new_results = await asyncio.gather(*tasks)
        stage1_results.extend(new_results)
        stage1_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stage1_path, "a", encoding="utf-8") as f:
            for r in new_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stage1_pass = [r for r in stage1_results if r["distill"]["any_correct"]]
    print(f"  Stage 1 pass: {len(stage1_pass)}/{len(stage1_results)} "
          f"({100*len(stage1_pass)//max(1,len(stage1_results))}%)")

    # Resume stage 2
    done_s2: set = set()
    stage2_results: List[Dict] = []
    if stage2_path.exists():
        with open(stage2_path, encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    r = json.loads(l)
                    stage2_results.append(r)
                    done_s2.add(r["source_id"])

    todo_s2 = [r for r in stage1_pass if r["source_id"] not in done_s2]
    print(f"  Stage 2: {len(todo_s2)} remaining rationales to generate")

    if todo_s2:
        tasks_s2 = [_stage2_example(ex, client, sem, api_key) for ex in todo_s2]
        new_s2 = await asyncio.gather(*tasks_s2)
        stage2_results.extend(new_s2)
        with open(stage2_path, "a", encoding="utf-8") as f:
            for r in new_s2:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stage2_valid = sum(1 for r in stage2_results if r["distill"]["rationale_valid"])
    print(f"  Stage 2 valid: {stage2_valid}/{len(stage2_results)} "
          f"({100*stage2_valid//max(1,len(stage2_results))}%)")

    return {
        "total":        len(records),
        "stage1_pass":  len(stage1_pass),
        "stage2_valid": stage2_valid,
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SUPPORTED_DATASETS = ["mera", "ceval", "milu", "global_mmlu"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets",    nargs="+", default=SUPPORTED_DATASETS,
                        choices=SUPPORTED_DATASETS)
    parser.add_argument("--sft_dir",     type=Path,
                        default=Path("multilingual/data/sft/v7"))
    parser.add_argument("--output_dir",  type=Path,
                        default=Path("multilingual/data/distilled/v7"))
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--dry_run",     action="store_true")
    return parser.parse_args()


async def main_async(args):
    api_key = os.environ.get("CSCS_SERVING_API", "")
    if not api_key:
        raise ValueError("CSCS_SERVING_API environment variable not set.")

    per_dir = args.sft_dir / "per_dataset"
    if not per_dir.exists():
        raise FileNotFoundError(f"per_dataset dir not found: {per_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    files_to_process = []
    for dataset in args.datasets:
        for path in sorted(per_dir.glob(f"{dataset}_*_train.jsonl")):
            files_to_process.append(path)

    if not files_to_process:
        print(f"No files found for {args.datasets} in {per_dir}")
        return

    print(f"Files to distill ({len(files_to_process)}):")
    for p in files_to_process:
        print(f"  {p.name}")

    if args.dry_run:
        print("\n[DRY RUN] Processing 5 examples per file.\n")

    sem = asyncio.Semaphore(args.concurrency)
    summary = []

    async with httpx.AsyncClient() as client:
        for src_path in files_to_process:
            base = src_path.stem.replace("_train", "")
            stage1_path = args.output_dir / f"{base}_stage1.jsonl"
            stage2_path = args.output_dir / f"{base}_stage2.jsonl"

            print(f"\n── {src_path.name} ──")
            t0 = time.time()
            stats = await _distill_file(
                src_path, stage1_path, stage2_path,
                client, sem, api_key, args.dry_run,
            )
            elapsed = time.time() - t0
            stats["file"]      = src_path.name
            stats["elapsed_s"] = round(elapsed, 1)
            summary.append(stats)
            print(f"  Elapsed: {elapsed:.1f}s")

    # Summary
    print("\n" + "="*60)
    print("DISTILLATION SUMMARY")
    print("="*60)
    total_ex = total_s1 = total_s2 = 0
    for s in summary:
        print(f"  {s['file']:50s}  "
              f"total={s['total']:4d}  "
              f"s1_pass={s['stage1_pass']:4d}  "
              f"s2_valid={s['stage2_valid']:4d}  "
              f"({s['elapsed_s']}s)")
        total_ex += s["total"]
        total_s1 += s["stage1_pass"]
        total_s2 += s["stage2_valid"]
    print(f"\n  TOTAL: {total_ex} → stage1 {total_s1} → stage2 {total_s2}")

    summary_path = args.output_dir / "distill_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary: {summary_path}")


def main():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()