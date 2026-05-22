"""multilingual/distillation/distill.py

Two-stage teacher distillation pipeline.

Stage 1: Two teachers answer independently with reasoning + boxed answer.
         Keep examples where at least one teacher matches gold.
         Only generate rationales when primary teacher (Qwen) got it right.

Stage 2: Qwen3.5-27B generates a short same-language rationale
         conditioned on the gold answer, only for high-confidence examples
         (primary_correct=True). If unsure, outputs NOT_SURE and is filtered out.

Confidence levels:
  high        — both teachers correct
  medium-high — primary correct, tiebreaker correct (verifier wrong)
  medium      — only verifier correct
  low         — all wrong

Rationale generation: only for high + medium-high (primary_correct=True).

Notes:
  - Qwen3.5-27B is a thinking model — responses include <think>...</think>.
    Extraction strips think blocks automatically.
  - Resumable: appends to existing output files, skips done source_ids.
  - Progress logged every LOG_EVERY examples per phase.

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
TEACHER_PRIMARY  = "Qwen/Qwen3.5-27B"               # main teacher + rationale generator
TEACHER_VERIFIER = "meta-llama/Llama-3.3-70B-Instruct"  # independent verifier
TEACHER_TIEBREAK = "swiss-ai/Apertus-70B-Instruct-2509"  # tie-breaker

MAX_TOKENS_THINKING    = 1024  # thinking models (Qwen, Apertus) — need room for <think>
MAX_TOKENS_NONTHINKING = 64    # Llama — non-thinking, answer only
MAX_TOKENS_RATIONALE   = 1200  # stage 2 rationale generation

DEFAULT_CONCURRENCY = 4
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 3.0
LOG_EVERY           = 25      # print progress every N examples

NOT_SURE_MARKER = "NOT_SURE"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

STAGE1_SYSTEM = (
    "You are carefully answering a multiple-choice or numeric question. "
    "Think step by step, then give your final answer inside \\boxed{}.\n"
    "Rules:\n"
    "- For multiple-choice questions: output the correct capital letter, e.g. \\boxed{B}\n"
    "- For numeric questions: output the final number, e.g. \\boxed{42}\n"
    "- Always end with exactly one \\boxed{} answer.\n"
    "- Never leave the answer unboxed."
)

STAGE2_SYSTEM = (
    "You are a multilingual teaching assistant. "
    "You will be given a multiple-choice question and its correct answer.\n"
    "Write a short explanation (2-4 sentences) in the SAME LANGUAGE as the question "
    "explaining why the given answer is correct.\n"
    "Rules:\n"
    "- Be concise and factual.\n"
    "- Do not repeat the question.\n"
    "- Respond in the same language as the question.\n"
    "- If you are not confident about the explanation, write only: NOT_SURE\n"
    "- End your response with the boxed answer on its own line: \\boxed{X}"
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
    """Single API call with retries. Returns raw content or None on failure."""
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
                timeout=120.0,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < retries - 1:
                await asyncio.sleep(DEFAULT_RETRY_DELAY * (attempt + 1))
            else:
                print(f"    [API error] {model}: {e}")
                return None
    return None

# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def _strip_think(text: str) -> str:
    """Remove <think>...</think> block from output."""
    if '</think>' in text:
        return text.split('</think>')[-1].strip()
    if re.search(r'</think>', text, re.IGNORECASE):
        return re.split(r'</think>', text, flags=re.IGNORECASE)[-1].strip()
    return text


def _extract_boxed(text: Optional[str]) -> Optional[str]:
    """Extract \\boxed{X} content, stripping think blocks first."""
    if not text:
        return None
    for src in (_strip_think(text), text):
        m = re.search(r'\\boxed\{([^}]+)\}', src)
        if m:
            return m.group(1).strip().upper()
    return None


def _extract_letter(text: Optional[str]) -> Optional[str]:
    """Fallback: extract a single A/B/C/D letter, handling think blocks."""
    if not text:
        return None
    cleaned = _strip_think(text).upper()
    if cleaned in ("A", "B", "C", "D"):
        return cleaned
    lines = [l.strip() for l in cleaned.splitlines() if l.strip()]
    if lines:
        last = lines[-1]
        if last in ("A", "B", "C", "D"):
            return last
        m = re.search(r'\b([A-D])\b', last)
        if m:
            return m.group(1)
    m = re.search(r'\b([A-D])\b', cleaned)
    return m.group(1) if m else None


def _extract_answer(text: Optional[str], task_type: str) -> Optional[str]:
    """Extract answer from model output — boxed first, letter fallback for MCQ."""
    boxed = _extract_boxed(text)
    if boxed:
        return boxed
    if task_type == "mcq":
        return _extract_letter(text)
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
    gold      = str(ex["answer"]).strip().upper()
    prompt    = ex["messages"][1]["content"]
    task_type = ex.get("task_type", "mcq")

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

    primary_ans  = _extract_answer(primary_raw,  task_type)
    verifier_ans = _extract_answer(verifier_raw, task_type)

    primary_correct  = (primary_ans  == gold)
    verifier_correct = (verifier_ans == gold)

    # Tie-breaker only when they disagree
    tiebreak_ans = None
    if primary_correct != verifier_correct:
        async with sem:
            tb_raw = await _call_api(
                client, TEACHER_TIEBREAK, messages_s1, api_key,
                max_tokens=MAX_TOKENS_THINKING,
            )
        tiebreak_ans = _extract_answer(tb_raw, task_type)

    # stage 1 passing criteria
    tiebreak_correct = (tiebreak_ans == gold)
    accepted_answer = primary_correct or verifier_correct or tiebreak_correct
    rationale_eligible = primary_correct and (verifier_correct or tiebreak_correct)
    if primary_correct and verifier_correct:
        confidence = "high"
    elif primary_correct and tiebreak_correct:
        confidence = "medium-high"
    elif verifier_correct or tiebreak_correct:
        confidence = "medium"
    else:
        confidence = "low"

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
            "tiebreak_correct":    tiebreak_correct,
            "accepted_answer":     accepted_answer,
            "rationale_eligible":  rationale_eligible,
            "confidence":          confidence,
        },
    }

# ---------------------------------------------------------------------------
# Stage 2: rationale generation
# ---------------------------------------------------------------------------

def _should_generate_rationale(r: Dict) -> bool:
    """Generate rationale only when Qwen is correct and independently confirmed."""
    return bool(r["distill"].get("rationale_eligible", False))


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

    rationale_clean = _strip_think(rationale_raw) if rationale_raw else None
    boxed           = _extract_boxed(rationale_raw)

    # Validity checks
    valid = False
    invalid_reason = "none"
    if not rationale_clean:
        invalid_reason = "empty"
    elif rationale_clean.strip().startswith(NOT_SURE_MARKER):
        invalid_reason = "not_sure"
    elif boxed != gold:
        invalid_reason = f"wrong_boxed:{boxed}"
    else:
        n_words = len(rationale_clean.split())
        if n_words <= 5:
            invalid_reason = "too_short"
        elif n_words >= 300:
            invalid_reason = "too_long"
        else:
            valid = True

    result = {
        **ex,
        "distill": {
            **ex["distill"],
            "rationale_raw":    rationale_raw,
            "rationale_clean":  rationale_clean,
            "rationale_valid":  valid,
            "rationale_boxed":  boxed,
            "invalid_reason":   invalid_reason if not valid else None,
        },
    }

    # Replace assistant message with clean rationale if valid
    if valid and rationale_clean:
        result["messages"] = list(ex["messages"])
        assistant_msg = { "role": "assistant", "content": rationale_clean}
        if len(result["messages"]) >= 3:
            result["messages"][2] = assistant_msg
        else:
            result["messages"].append(assistant_msg)

# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

class _Progress:
    """Simple counter for logging progress every LOG_EVERY examples."""

    def __init__(self, total: int, phase: str, log_every: int = LOG_EVERY):
        self.total    = total
        self.phase    = phase
        self.log_every = log_every
        self.done     = 0
        self.correct  = 0
        self.t0       = time.time()

    def update(self, correct: bool = True) -> None:
        self.done += 1
        if correct:
            self.correct += 1
        if self.done % self.log_every == 0 or self.done == self.total:
            elapsed = time.time() - self.t0
            rate    = self.done / elapsed if elapsed > 0 else 0
            eta     = (self.total - self.done) / rate if rate > 0 else 0
            pct     = 100 * self.correct // max(1, self.done)
            print(f"    [{self.phase}] {self.done}/{self.total} "
                  f"| correct {self.correct} ({pct}%) "
                  f"| {rate:.1f} ex/s "
                  f"| ETA {eta:.0f}s")

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
    log_every: int,
    subset_filter: Optional[str] = None,
) -> Dict[str, Any]:
    with open(src_path, encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]

    records = [r for r in records
               if r.get("split") == "train" and r.get("task_type") == "mcq"]

    # MERA subset filter: mera_mamuramu / mera_mathlogicqa
    if subset_filter:
        records = [r for r in records
                   if r.get("metadata", {}).get("subset") == subset_filter]

    if dry_run:
        records = records[:5]

    if not records:
        return {"total": 0, "stage1_pass": 0, "stage2_eligible": 0, "stage2_valid": 0}

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    done_ids: set       = set()
    stage1_results: List[Dict] = []
    if stage1_path.exists():
        with open(stage1_path, encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    r = json.loads(l)
                    stage1_results.append(r)
                    done_ids.add(r["source_id"])
        print(f"  Resuming stage 1: {len(done_ids)} already done")

    todo = [r for r in records if r["source_id"] not in done_ids]
    print(f"  Stage 1: {len(todo)} remaining / {len(records)} total")

    if todo:
        prog = _Progress(len(todo), "stage1", log_every)

        async def _s1_tracked(ex):
            result = await _stage1_example(ex, client, sem, api_key)
            prog.update(correct=result["distill"]["accepted_answer"])
            return result

        tasks      = [_s1_tracked(ex) for ex in todo]
        new_results = await asyncio.gather(*tasks)
        stage1_results.extend(new_results)
        stage1_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stage1_path, "a", encoding="utf-8") as f:
            for r in new_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stage1_pass = [r for r in stage1_results if r["distill"].get("accepted_answer", r["distill"].get("any_correct", False))]
    stage2_eligible = [r for r in stage1_pass if _should_generate_rationale(r)]

    conf_counts = {}
    for r in stage1_results:
        c = r["distill"]["confidence"]
        conf_counts[c] = conf_counts.get(c, 0) + 1

    print(f"  Stage 1 accepted: {len(stage1_pass)}/{len(stage1_results)} "
        f"({100*len(stage1_pass)//max(1,len(stage1_results))}%)")
    print(f"  Confidence breakdown: {conf_counts}")
    print(f"  Stage 2 eligible (rationale_eligible): {len(stage2_eligible)}")

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    done_s2: set        = set()
    stage2_results: List[Dict] = []
    if stage2_path.exists():
        with open(stage2_path, encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    r = json.loads(l)
                    stage2_results.append(r)
                    done_s2.add(r["source_id"])

    todo_s2 = [r for r in stage2_eligible if r["source_id"] not in done_s2]
    print(f"  Stage 2: {len(todo_s2)} remaining rationales to generate")

    if todo_s2:
        prog2 = _Progress(len(todo_s2), "stage2", log_every)

        async def _s2_tracked(ex):
            result = await _stage2_example(ex, client, sem, api_key)
            prog2.update(correct=result["distill"]["rationale_valid"])
            return result

        tasks_s2  = [_s2_tracked(ex) for ex in todo_s2]
        new_s2    = await asyncio.gather(*tasks_s2)
        stage2_results.extend(new_s2)
        with open(stage2_path, "a", encoding="utf-8") as f:
            for r in new_s2:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stage2_valid = sum(1 for r in stage2_results if r["distill"]["rationale_valid"])

    # Invalid reason breakdown
    invalid_reasons: Dict[str, int] = {}
    for r in stage2_results:
        reason = r["distill"].get("invalid_reason")
        if reason:
            invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1

    print(f"  Stage 2 valid: {stage2_valid}/{len(stage2_results)} "
          f"({100*stage2_valid//max(1,len(stage2_results))}%)")
    if invalid_reasons:
        print(f"  Invalid reasons: {invalid_reasons}")

    return {
        "total":           len(records),
        "stage1_pass":     len(stage1_pass),
        "stage2_eligible": len(stage2_eligible),
        "stage2_valid":    stage2_valid,
        "conf_counts":     conf_counts,
        "invalid_reasons": invalid_reasons,
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SUPPORTED_DATASETS = ["mera", "ceval", "milu", "global_mmlu",
                      "mera_mamuramu", "mera_mathlogicqa"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets",    nargs="+", default=SUPPORTED_DATASETS,
                        choices=SUPPORTED_DATASETS)
    parser.add_argument("--sft_dir",     type=Path,
                        default=Path("multilingual/data/sft/v7"))
    parser.add_argument("--output_dir",  type=Path,
                        default=Path("multilingual/data/distilled/v7"))
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--log_every",   type=int, default=LOG_EVERY,
                        help="Print progress every N examples per phase.")
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

    # Build list of (src_path, subset_filter, output_base)
    jobs: list = []
    for dataset in args.datasets:
        if dataset == "mera_mamuramu":
            for path in sorted(per_dir.glob("mera_*_train.jsonl")):
                jobs.append((path, "mamuramu", "mera_russian_mamuramu"))
        elif dataset == "mera_mathlogicqa":
            for path in sorted(per_dir.glob("mera_*_train.jsonl")):
                jobs.append((path, "mathlogicqa", "mera_russian_mathlogicqa"))
        else:
            for path in sorted(per_dir.glob(f"{dataset}_*_train.jsonl")):
                jobs.append((path, None, path.stem.replace("_train", "")))

    if not jobs:
        print(f"No files found for {args.datasets} in {per_dir}")
        return

    files_to_process = jobs  # rename for print
    print(f"Jobs to distill ({len(jobs)}):")
    for path, subset, base in jobs:
        tag = f" [subset={subset}]" if subset else ""
        print(f"  {path.name}{tag} → {base}")

    if args.dry_run:
        print("\n[DRY RUN] Processing 5 examples per file.\n")

    sem     = asyncio.Semaphore(args.concurrency)
    summary = []

    async with httpx.AsyncClient() as client:
        for src_path, subset_filter, base in jobs:
            stage1_path = args.output_dir / f"{base}_stage1.jsonl"
            stage2_path = args.output_dir / f"{base}_stage2.jsonl"

            tag = f" [subset={subset_filter}]" if subset_filter else ""
            print(f"\n── {src_path.name}{tag} ──")
            t0    = time.time()
            stats = await _distill_file(
                src_path, stage1_path, stage2_path,
                client, sem, api_key, args.dry_run, args.log_every,
                subset_filter=subset_filter,
            )
            elapsed         = time.time() - t0
            stats["file"]   = src_path.name
            stats["elapsed_s"] = round(elapsed, 1)
            summary.append(stats)
            print(f"  Done in {elapsed:.1f}s")

    # Summary
    print("\n" + "="*60)
    print("DISTILLATION SUMMARY")
    print("="*60)
    total_ex = total_s1 = total_s2e = total_s2v = 0
    for s in summary:
        print(f"  {s['file']:50s}  "
              f"total={s['total']:4d}  "
              f"s1_pass={s['stage1_pass']:4d}  "
              f"s2_eligible={s['stage2_eligible']:4d}  "
              f"s2_valid={s['stage2_valid']:4d}  "
              f"({s['elapsed_s']}s)")
        total_ex  += s["total"]
        total_s1  += s["stage1_pass"]
        total_s2e += s["stage2_eligible"]
        total_s2v += s["stage2_valid"]
    print(f"\n  TOTAL: {total_ex} → s1_pass {total_s1} "
          f"→ s2_eligible {total_s2e} → s2_valid {total_s2v}")

    summary_path = args.output_dir / "distill_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary: {summary_path}")


def main():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()