"""multilingual/distillation/distill_2stage.py

Two-stage teacher distillation pipeline.

Stage 1: Two teachers answer independently.
  - Primary verifier: Llama-3.3-70B (fast, non-thinking)
  - Fallback verifier: Apertus-70B (only called when Llama fails)
  - Qwen dropped from Stage 1 — too slow, causes rate limiting.
  - Acceptance rule: at least one model correct AND confirmed by another.

Stage 2: Rationale generation.
  - Use Llama when Llama was correct.
  - Use Apertus when Llama failed but Apertus confirmed.
  - Never use Qwen for rationale generation.

Saving: results are saved every SAVE_EVERY examples (not just at end),
so progress is not lost if the job dies.

Resumable: appends to existing output files, skips done source_ids.

Usage:
    export CSCS_SERVING_API="sk-rc-..."

    # Dry run (5 examples)
    python -m multilingual.distillation.distill_2stage \
        --datasets mera_mamuramu \
        --processed_dir multilingual/data/processed \
        --output_dir multilingual/data/distilled/v7_final \
        --dry_run

    # Full run
    python -m multilingual.distillation.distill_2stage \
        --datasets ceval milu mera_mamuramu \
        --processed_dir multilingual/data/processed \
        --output_dir multilingual/data/distilled/v7_final \
        --concurrency 3 \
        --log_every 25 \
        --save_every 25
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
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

# Two-teacher setup — Qwen dropped to reduce rate limiting
TEACHER_VERIFIER = "meta-llama/Llama-3.3-70B-Instruct"   # primary, fast
TEACHER_TIEBREAK = "swiss-ai/Apertus-70B-Instruct-2509"  # fallback

MAX_TOKENS_VERIFIER  = 256   # Llama — non-thinking
MAX_TOKENS_TIEBREAK  = 512   # Apertus — semi-thinking
MAX_TOKENS_RATIONALE = 1024  # stage 2

DEFAULT_CONCURRENCY = 3
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 5.0
LOG_EVERY  = 25
SAVE_EVERY = 25  # save to disk every N examples processed

NOT_SURE_MARKER = "NOT_SURE"

# Maps dataset name → list of (relative_path, output_base_name)
PROCESSED_INPUTS: Dict[str, List[tuple]] = {
    "ceval": [
        ("ceval/ceval_chinese_regional.jsonl",          "ceval_chinese_regional"),
        ("ceval/ceval_chinese_regional_implicit.jsonl", "ceval_chinese_regional_implicit"),
    ],
    "milu": [
        ("milu/milu_hindi_train_regional.jsonl",          "milu_hindi_regional"),
        ("milu/milu_hindi_train_regional_implicit.jsonl", "milu_hindi_regional_implicit"),
    ],
    "mera_mamuramu": [
        ("mera/mera_clean_mamuramu.jsonl", "mera_russian_mamuramu"),
    ],
}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

STAGE1_SYSTEM = (
    "You are carefully answering a multiple-choice question. "
    "Think step by step, then give your final answer inside \\boxed{}.\n"
    "Rules:\n"
    "- Output the correct capital letter inside \\boxed{}, e.g. \\boxed{B}\n"
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
    "- Write only in the same language as the question.\n"
    "- Do not include headings, bullet points, drafts, or meta-analysis.\n"
    "- If you are not confident, write only: NOT_SURE\n"
    "- End your response with the correct answer on its own line: \\boxed{X}"
)

LANGUAGE_NAMES = {
    "russian": "Russian", "chinese": "Chinese", "hindi": "Hindi",
    "italian": "Italian", "spanish": "Spanish", "english": "English",
}


def _lang(ex: Dict) -> str:
    return LANGUAGE_NAMES.get(str(ex.get("language", "")).lower(), "the same language as the question")


def _stage1_user(prompt: str, language: str) -> str:
    return (
        f"The question is written in {language}. "
        f"Reason in {language} and give your final boxed answer.\n\n{prompt}"
    )


def _stage2_user(prompt: str, gold: str, language: str) -> str:
    return (
        f"The question is written in {language}. "
        f"Write only the final explanation in {language}, not your reasoning process.\n\n"
        f"{prompt}\n\n"
        f"The correct answer is {gold}. "
        f"Write 2-4 short sentences in {language} explaining why this answer is correct. "
        f"No headings, no bullet points, no drafts. "
        f"Then write \\boxed{{{gold}}} on the last line."
    )

# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------

async def _call_api(
    client: httpx.AsyncClient,
    model: str,
    messages: List[Dict],
    api_key: str,
    max_tokens: int,
    temperature: float = 0.0,
    retries: int = DEFAULT_MAX_RETRIES,
) -> Optional[str]:
    payload = {
        "model": model, "messages": messages,
        "max_tokens": max_tokens, "temperature": temperature, "stream": False,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for attempt in range(retries):
        try:
            resp = await client.post(
                f"{API_BASE}/chat/completions",
                json=payload, headers=headers, timeout=120.0,
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
# Extraction helpers
# ---------------------------------------------------------------------------

def _strip_think(text: str) -> str:
    if '</think>' in text:
        return text.split('</think>')[-1].strip()
    if re.search(r'</think>', text, re.IGNORECASE):
        return re.split(r'</think>', text, flags=re.IGNORECASE)[-1].strip()
    return text


def _extract_boxed(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    for src in (_strip_think(text), text):
        matches = re.findall(r'\\boxed\{([^}]+)\}', src)
        if matches:
            return matches[-1].strip().upper()
    return None


def _extract_letter(text: Optional[str]) -> Optional[str]:
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


def _extract_answer(text: Optional[str]) -> Optional[str]:
    """Try boxed first, fall back to single letter."""
    return _extract_boxed(text) or _extract_letter(text)


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _too_short(text: str) -> bool:
    return (len(text) <= 20) if _contains_cjk(text) else (len(text.split()) <= 4)


def _too_long(text: str) -> bool:
    return (len(text) >= 900) if _contains_cjk(text) else (len(text.split()) >= 300)

# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------

async def _stage1_example(
    ex: Dict,
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    api_key: str,
) -> Dict:
    gold      = str(ex["answer"]).strip().upper()
    prompt    = ex["messages"][1]["content"]
    language  = _lang(ex)

    messages_s1 = [
        {"role": "system", "content": STAGE1_SYSTEM},
        {"role": "user",   "content": _stage1_user(prompt, language)},
    ]

    # Always call Llama (fast)
    async with sem:
        verifier_raw = await _call_api(
            client, TEACHER_VERIFIER, messages_s1, api_key,
            max_tokens=MAX_TOKENS_VERIFIER,
        )

    verifier_ans     = _extract_answer(verifier_raw)
    verifier_correct = (verifier_ans == gold)

    # Only call Apertus if Llama failed
    tiebreak_raw     = None
    tiebreak_ans     = None
    tiebreak_correct = False

    if not verifier_correct:
        async with sem:
            tiebreak_raw = await _call_api(
                client, TEACHER_TIEBREAK, messages_s1, api_key,
                max_tokens=MAX_TOKENS_TIEBREAK,
            )
        tiebreak_ans     = _extract_answer(tiebreak_raw)
        tiebreak_correct = (tiebreak_ans == gold)

    # Acceptance: Llama correct, OR Apertus confirms when Llama failed
    accepted = verifier_correct or tiebreak_correct

    if verifier_correct:
        confidence             = "high"
        rationale_teacher      = TEACHER_VERIFIER
        rationale_source       = "verifier"
    elif tiebreak_correct:
        confidence             = "medium"
        rationale_teacher      = TEACHER_TIEBREAK
        rationale_source       = "tiebreak"
    else:
        confidence             = "low"
        rationale_teacher      = None
        rationale_source       = None

    return {
        **ex,
        "distill": {
            "gold":              gold,
            "verifier_raw":      verifier_raw,
            "tiebreak_raw":      tiebreak_raw,
            "verifier_ans":      verifier_ans,
            "tiebreak_ans":      tiebreak_ans,
            "verifier_correct":  verifier_correct,
            "tiebreak_correct":  tiebreak_correct,
            "accepted":          accepted,
            "confidence":        confidence,
            "rationale_teacher": rationale_teacher,
            "rationale_source":  rationale_source,
        },
    }

# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------

async def _stage2_example(
    ex: Dict,
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    api_key: str,
) -> Dict:
    gold     = ex["distill"]["gold"]
    prompt   = ex["messages"][1]["content"]
    language = _lang(ex)
    teacher  = ex["distill"]["rationale_teacher"]

    if not teacher:
        raise ValueError(f"No rationale teacher for {ex.get('source_id')}")

    messages_s2 = [
        {"role": "system", "content": STAGE2_SYSTEM},
        {"role": "user",   "content": _stage2_user(prompt, gold, language)},
    ]

    async with sem:
        raw = await _call_api(
            client, teacher, messages_s2, api_key,
            max_tokens=MAX_TOKENS_RATIONALE, temperature=0.0,
        )

    clean = _strip_think(raw) if raw else None
    boxed = _extract_boxed(raw)
    expected_final = f"\\boxed{{{gold}}}"

    valid         = False
    invalid_reason = "none"
    if not clean:
        invalid_reason = "empty"
    elif clean.strip().startswith(NOT_SURE_MARKER):
        invalid_reason = "not_sure"
    elif boxed != gold:
        invalid_reason = f"wrong_boxed:{boxed}"
    elif _too_short(clean):
        invalid_reason = "too_short"
    elif _too_long(clean):
        invalid_reason = "too_long"
    elif not clean.strip().endswith(expected_final):
        invalid_reason = "not_ending_with_boxed"
    else:
        valid = True

    result = {
        **ex,
        "distill": {
            **ex["distill"],
            "rationale_raw":    raw,
            "rationale_clean":  clean,
            "rationale_valid":  valid,
            "rationale_boxed":  boxed,
            "invalid_reason":   None if valid else invalid_reason,
        },
    }

    if valid and clean:
        result["messages"] = list(ex["messages"])
        assistant_msg = {"role": "assistant", "content": clean}
        if len(result["messages"]) >= 3:
            result["messages"][2] = assistant_msg
        else:
            result["messages"].append(assistant_msg)

    return result

# ---------------------------------------------------------------------------
# Progress + saving
# ---------------------------------------------------------------------------

class _Progress:
    def __init__(self, total: int, phase: str, log_every: int):
        self.total, self.phase, self.log_every = total, phase, log_every
        self.done = self.correct = 0
        self.t0 = time.time()

    def update(self, correct: bool) -> None:
        self.done += 1
        if correct:
            self.correct += 1
        if self.done % self.log_every == 0 or self.done == self.total:
            elapsed = time.time() - self.t0
            rate    = self.done / elapsed if elapsed > 0 else 0
            eta     = (self.total - self.done) / rate if rate > 0 else 0
            pct     = 100 * self.correct // max(1, self.done)
            print(f"    [{self.phase}] {self.done}/{self.total} "
                  f"| ok={self.correct} ({pct}%) "
                  f"| {rate:.1f} ex/s | ETA {eta:.0f}s")


def _append_jsonl(records: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _load_done_ids(path: Path) -> tuple[set, List[Dict]]:
    """Load existing results and return (done_source_ids, records)."""
    done_ids, records = set(), []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    records.append(r)
                    done_ids.add(r["source_id"])
    return done_ids, records

# ---------------------------------------------------------------------------
# Record normalization
# ---------------------------------------------------------------------------

def _make_prompt(r: Dict) -> str:
    if r.get("prompt"):
        return str(r["prompt"])
    if r.get("messages") and len(r["messages"]) >= 2:
        return str(r["messages"][1]["content"])
    q = str(r.get("question", "")).strip()
    for keys in [("A","B","C","D"), ("option1","option2","option3","option4")]:
        if all(k in r for k in keys):
            letters = "ABCD"
            opts = "\n".join(f"{letters[i]}) {r[k]}" for i, k in enumerate(keys))
            return f"{q}\n\n{opts}"
    raise ValueError(f"Cannot build prompt for {r.get('id', r.get('source_id', '?'))}")


def _normalize(r: Dict, src_path: Path, idx: int) -> Dict:
    prompt    = _make_prompt(r)
    answer    = str(r["answer"]).strip().upper()
    source_id = str(r.get("source_id") or r.get("id") or f"{src_path.stem}_{idx:06d}")
    messages  = r.get("messages") or [
        {"role": "system",    "content": ""},
        {"role": "user",      "content": prompt},
        {"role": "assistant", "content": f"\\boxed{{{answer}}}"},
    ]
    return {
        **r,
        "source_id": source_id,
        "answer":    answer,
        "task_type": r.get("task_type", "mcq"),
        "split":     "train",
        "messages":  messages,
        "metadata":  dict(r.get("metadata", {})),
    }


def _load_processed(src_path: Path) -> List[Dict]:
    with open(src_path, encoding="utf-8") as f:
        raw = [json.loads(l) for l in f if l.strip()]
    return [_normalize(r, src_path, i) for i, r in enumerate(raw)]

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
    save_every: int,
    debug_n: int = 5,
) -> Dict[str, Any]:
    records = _load_processed(src_path)
    records = [r for r in records if r.get("task_type") == "mcq"]

    if dry_run:
        records = random.Random(42).sample(records, min(debug_n, len(records)))

    if not records:
        return {"total": 0, "stage1_pass": 0, "stage2_eligible": 0, "stage2_valid": 0}

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    done_ids, stage1_results = _load_done_ids(stage1_path)
    if done_ids:
        print(f"  Resuming stage 1: {len(done_ids)} already done")

    todo = [r for r in records if r["source_id"] not in done_ids]
    print(f"  Stage 1: {len(todo)} remaining / {len(records)} total")

    if todo:
        prog    = _Progress(len(todo), "stage1", log_every)
        buffer  = []

        async def _s1(ex):
            result = await _stage1_example(ex, client, sem, api_key)
            prog.update(correct=result["distill"]["accepted"])
            return result

        # Process in save_every batches
        for batch_start in range(0, len(todo), save_every):
            batch   = todo[batch_start:batch_start + save_every]
            results = await asyncio.gather(*[_s1(ex) for ex in batch])
            stage1_results.extend(results)
            _append_jsonl(results, stage1_path)

    stage1_pass     = [r for r in stage1_results if r["distill"]["accepted"]]
    stage2_eligible = [r for r in stage1_pass if r["distill"]["rationale_teacher"]]

    conf_counts = {}
    for r in stage1_results:
        c = r["distill"]["confidence"]
        conf_counts[c] = conf_counts.get(c, 0) + 1

    src_counts = {}
    for r in stage2_eligible:
        s = r["distill"].get("rationale_source", "none")
        src_counts[s] = src_counts.get(s, 0) + 1

    print(f"  Stage 1 accepted: {len(stage1_pass)}/{len(stage1_results)} "
          f"({100*len(stage1_pass)//max(1,len(stage1_results))}%)")
    print(f"  Confidence: {conf_counts}")
    print(f"  Stage 2 eligible: {len(stage2_eligible)} | sources: {src_counts}")

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    done_s2, stage2_results = _load_done_ids(stage2_path)
    todo_s2 = [r for r in stage2_eligible if r["source_id"] not in done_s2]
    print(f"  Stage 2: {len(todo_s2)} remaining rationales")

    if todo_s2:
        prog2 = _Progress(len(todo_s2), "stage2", log_every)

        async def _s2(ex):
            result = await _stage2_example(ex, client, sem, api_key)
            prog2.update(correct=result["distill"]["rationale_valid"])
            return result

        for batch_start in range(0, len(todo_s2), save_every):
            batch   = todo_s2[batch_start:batch_start + save_every]
            results = await asyncio.gather(*[_s2(ex) for ex in batch])
            stage2_results.extend(results)
            _append_jsonl(results, stage2_path)

    stage2_valid = sum(1 for r in stage2_results if r["distill"]["rationale_valid"])
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
        "src_counts":      src_counts,
        "invalid_reasons": invalid_reasons,
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SUPPORTED_DATASETS = list(PROCESSED_INPUTS.keys())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets",      nargs="+", default=SUPPORTED_DATASETS, choices=SUPPORTED_DATASETS)
    p.add_argument("--processed_dir", type=Path, default=Path("multilingual/data/processed"))
    p.add_argument("--output_dir",    type=Path, default=Path("multilingual/data/distilled/v7_final"))
    p.add_argument("--concurrency",   type=int,  default=DEFAULT_CONCURRENCY)
    p.add_argument("--log_every",     type=int,  default=LOG_EVERY)
    p.add_argument("--save_every",    type=int,  default=SAVE_EVERY,
                   help="Save to disk every N examples (prevents losing progress on failure).")
    p.add_argument("--dry_run",       action="store_true")
    p.add_argument("--debug_n",       type=int,  default=5)
    return p.parse_args()


def _build_jobs(processed_dir: Path, datasets: List[str]) -> List[tuple]:
    jobs = []
    for dataset in datasets:
        for rel_path, output_base in PROCESSED_INPUTS[dataset]:
            path = processed_dir / rel_path
            if not path.exists():
                raise FileNotFoundError(f"Processed file not found: {path}")
            jobs.append((path, output_base))
    return jobs


async def main_async(args):
    api_key = os.environ.get("CSCS_SERVING_API", "")
    if not api_key:
        raise ValueError("CSCS_SERVING_API not set.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = _build_jobs(args.processed_dir, args.datasets)

    print(f"Jobs ({len(jobs)}):")
    for path, base in jobs:
        print(f"  {path.name} → {base}")
    if args.dry_run:
        print(f"\n[DRY RUN] {args.debug_n} examples per file.\n")

    sem     = asyncio.Semaphore(args.concurrency)
    summary = []

    async with httpx.AsyncClient() as client:
        for src_path, base in jobs:
            stage1_path = args.output_dir / f"{base}_stage1.jsonl"
            stage2_path = args.output_dir / f"{base}_stage2.jsonl"

            print(f"\n── {src_path.name} ──")
            t0    = time.time()
            stats = await _distill_file(
                src_path, stage1_path, stage2_path,
                client, sem, api_key,
                args.dry_run, args.log_every, args.save_every, args.debug_n,
            )
            elapsed          = time.time() - t0
            stats["file"]    = src_path.name
            stats["elapsed"] = round(elapsed, 1)
            summary.append(stats)
            print(f"  Done in {elapsed:.1f}s")

    print("\n" + "="*60)
    print("DISTILLATION SUMMARY")
    print("="*60)
    tot_ex = tot_s1 = tot_s2e = tot_s2v = 0
    for s in summary:
        print(f"  {s['file']:50s}  "
              f"total={s['total']:4d}  s1={s['stage1_pass']:4d}  "
              f"s2_elig={s['stage2_eligible']:4d}  s2_valid={s['stage2_valid']:4d}  "
              f"({s['elapsed']}s)")
        tot_ex  += s["total"]
        tot_s1  += s["stage1_pass"]
        tot_s2e += s["stage2_eligible"]
        tot_s2v += s["stage2_valid"]
    print(f"\n  TOTAL: {tot_ex} → s1 {tot_s1} → s2_elig {tot_s2e} → s2_valid {tot_s2v}")

    summary_path = args.output_dir / "distill_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary: {summary_path}")


def main():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()