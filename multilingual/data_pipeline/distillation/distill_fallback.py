"""multilingual/data_pipeline/distillation/distill_fallback.py

Teacher distillation pipeline for multilingual MCQ SFT data.

This script reads processed MCQ datasets from `multilingual/data/processed`,
queries one or more teacher models, validates the generated rationale, and
writes distilled examples that can later be used for supervised fine-tuning.

Per example:
  1. Try Llama independently up to 4 times.
     - Attempt 1 uses temperature 0.0.
     - Attempts 2-4 use the CLI sampling temperature and top_p.
  2. If Llama never gives a valid correct answer, try Apertus independently 2 times.
  3. If both teachers fail independently, give the gold answer to Llama and ask for a rationale.
  4. If Llama's gold-answer rationale is invalid, try the same gold-answer fallback with Apertus.
  5. Keep only valid rationales ending in the correct \\boxed{LETTER}; otherwise discard.

Saving is append-only and resumable:
    <output_dir>/<base>_distilled.jsonl  ← valid examples only, for SFT
    <output_dir>/<base>_all.jsonl        ← all attempts, for analysis/resuming
    <output_dir>/<base>_fail.jsonl       ← failed examples

Usage:
    export CSCS_SERVING_API="sk-rc-..."

    python -m multilingual.data_pipeline.distillation.distill_fallback \\
        --datasets ceval milu mera_mamuramu \\
        --processed_dir multilingual/data/processed \\
        --output_dir multilingual/data/distilled/v1stage_gold_fallback \\
        --concurrency 4 \\
        --sample_temperature 0.25 \\
        --top_p 0.95 \\
        --save_every 25

Dry run:
    python -m multilingual.data_pipeline.distillation.distill_fallback \\
        --datasets mera_mamuramu \\
        --processed_dir multilingual/data/processed \\
        --output_dir multilingual/data/distilled/debug_fallback \\
        --dry_run
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

# ── Teacher config ──────────────────────────────────────────────────────────
# To use Qwen as primary: set TEACHER_PRIMARY = "Qwen/Qwen3.5-27B"
# and USE_NO_THINK_PRIMARY = True (skips Qwen's think block via /no_think)
TEACHER_PRIMARY   = "meta-llama/Llama-3.3-70B-Instruct"
TEACHER_FALLBACK  = "swiss-ai/Apertus-70B-Instruct-2509"
N_ADDITIONAL_RETRIES_MAIN = 3
N_ADDITIONAL_RETRIES_FALLBACK = 0 # just one try from fallback if we set this to 0

USE_NO_THINK_PRIMARY  = False  # set True if TEACHER_PRIMARY is Qwen
USE_NO_THINK_FALLBACK = False  # set True if TEACHER_FALLBACK is Qwen

MAX_TOKENS_PRIMARY  = 512   # Llama: short rationale, no think
MAX_TOKENS_FALLBACK = 512   # Apertus: same
# If using Qwen: set MAX_TOKENS_PRIMARY = 2048 (needs room for think block)
# ───────────────────────────────────────────────────────────────────────────

DEFAULT_CONCURRENCY = 4
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0
LOG_EVERY  = 25
SAVE_EVERY = 25
DEFAULT_SAMPLE_TEMPERATURE = 0.25
DEFAULT_TOP_P = 0.95

PROCESSED_INPUTS: Dict[str, List[tuple]] = {
    "ceval": [
        ("ceval/ceval_chinese_regional.jsonl",          "ceval_chinese_regional"),
        ("ceval/ceval_chinese_regional_implicit.jsonl", "ceval_chinese_regional_implicit"),
    ],
    "milu": [
        ("milu/milu_hindi_train_regional.jsonl",          "milu_hindi_regional"),
        ("milu/milu_hindi_train_regional_implicit.jsonl", "milu_hindi_regional_implicit"),
    ],
    "global_mmlu_regional": [
        ("global_mmlu/global_mmlu_russian_regional.jsonl",  "global_mmlu_russian_regional"),
        ("global_mmlu/global_mmlu_chinese_regional.jsonl",  "global_mmlu_chinese_regional"),
        ("global_mmlu/global_mmlu_italian_regional.jsonl",  "global_mmlu_italian_regional"),
        ("global_mmlu/global_mmlu_spanish_regional.jsonl",  "global_mmlu_spanish_regional"),
        ("global_mmlu/global_mmlu_hindi_regional.jsonl",    "global_mmlu_hindi_regional"),
    ],
    "include": [
        ("include/include_russian_all.jsonl",  "include_russian_all"),
        ("include/include_chinese_all.jsonl",  "include_chinese_all"),
        ("include/include_italian_all.jsonl",  "include_italian_all"),
        ("include/include_spanish_all.jsonl",  "include_spanish_all"),
        ("include/include_hindi_all.jsonl",    "include_hindi_all"),
    ],
    "cultsportqa": [
        ("cultsportqa/cultsportqa_hindi_all.jsonl",   "cultsportqa_hindi"),
        ("cultsportqa/cultsportqa_chinese_all.jsonl", "cultsportqa_chinese"),
    ],
    "head_qa_spanish": [
        ("head_qa/head_qa_spanish_all.jsonl", "head_qa_spanish"),
    ],
    "head_qa_italian": [
        ("head_qa/head_qa_italian_all.jsonl", "head_qa_italian"),
    ],
    # new
    "mera": [
        ("mera/mera_clean_mamuramu.jsonl", "mera_russian_mamuramu"),
        ("mera/mera_clean_ruopenbookqa.jsonl", "mera_russian_ruopenbookqa"), # new
        ("mera/mera_clean_ruworldtree.jsonl", "mera_russian_ruworldtree"), # this too
    ],
    "global_piqa": [
        ("global_piqa/global_piqa_italian_all.jsonl",  "global_piqa_italian"),
        ("global_piqa/global_piqa_spanish_all.jsonl",  "global_piqa_spanish"),
        ("global_piqa/global_piqa_hindi_all.jsonl",    "global_piqa_hindi"),
        ("global_piqa/global_piqa_chinese_all.jsonl",  "global_piqa_chinese"),
        ("global_piqa/global_piqa_russian_all.jsonl",  "global_piqa_russian"),
    ],
    "exams": [
        ("exams/exams_italian_all.jsonl", "exams_italian"),
        ("exams/exams_spanish_all.jsonl", "exams_spanish"),
    ],
    "global_mmlu_region_agnostic": [
        ("global_mmlu/global_mmlu_russian_shuffled_region_agnostic.jsonl",  "global_mmlu_russian_region_agnostic"),
        # ("global_mmlu/global_mmlu_chinese_shuffled_region_agnostic.jsonl",  "global_mmlu_chinese_region_agnostic"),
        ("global_mmlu/global_mmlu_italian_shuffled_region_agnostic.jsonl",  "global_mmlu_italian_region_agnostic"),
        ("global_mmlu/global_mmlu_spanish_shuffled_region_agnostic.jsonl",  "global_mmlu_spanish_region_agnostic"),
        # ("global_mmlu/global_mmlu_hindi_shuffled_region_agnostic.jsonl",    "global_mmlu_hindi_region_agnostic"),
    ],
    "italian_civil_code": [
        ("italian_civil_code/italian_civil_code.jsonl", "italian_civil_code"),
    ],
    "spain_boe_legislation": [
        ("spain_boe_legislation/spain_boe_legislation_generated.jsonl", "spain_boe_legislation"),
    ],
    "polizia_di_stato": [
        ("polizia_di_stato/polizia_di_stato_clean.jsonl", "polizia_di_stato"),
    ],
    "quaderno_prefettizia": [
        ("quaderno_prefettizia/quaderno_prefettizia_clean.jsonl", "quaderno_prefettizia"),
        ("quaderno_prefettizia/quaderno_prefettizia_economia_politica_clean.jsonl", "quaderno_prefettizia_economia_politica"),
    ],
}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

LANGUAGE_NAMES = {
    "russian": "Russian", "chinese": "Chinese", "hindi": "Hindi",
    "italian": "Italian", "spanish": "Spanish", "english": "English",
}

SYSTEM_MESSAGE = """You are a concise assistant solving multilingual regional knowledge MCQ problems written in {language}. 
Your goal is to help a small student model learn reasoning, so you must write clear, explicit, and non-verbose answers.
   Rules:
   1. Write a clear and concise reasoning in {language}.
      - Identify the key fact, rule, or concept needed to answer the question.
      - Explain briefly why the selected option follows from it.
      - Do not copy or paraphrase the whole question.
      - Avoid generic introductions, speculation, and unnecessary details.
      - Prefer 2-5 concise sentences unless the question genuinely requires more.
      - Mention wrong options only when they are plausible or when contrasting them clarifies the answer.
   2. At the end of your reasoning, you must include your answer in \\boxed{{LETTER}}.
      - For example, if the answer is A, you must end your response with \\boxed{{A}}.
      - Do not add any extra text after the boxed answer.
   If you are not confident that your explanation is factually correct, write only: NOT_SURE.
   Do not output anything else besides the reasoning and the \\boxed{{LETTER}} at the end."""


SYSTEM_MESSAGE_FALLBACK = """You are a concise assistant solving multilingual regional knowledge MCQ problems in {language}.
Your goal is to help a small student model learn reasoning, so you must write clear, explicit, and non-verbose answers.
You are given the correct answer to a question. Explain why it is right, as if you solved it independently.
The question is expected to be valid, but rare corrupted, non-sensical examples may appear.

Rules:
1. Write only FAIL if the question, options, or provided answer appear corrupted or inconsistent, for example:
   - the correct option is malformed, empty, unrelated, or the wrong answer type;
   - the question and/or options are in the wrong language
   - options contain timestamps, parsed dates, placeholders, duplicated broken text, or nonsensical choices;
   - the provided answer contradicts the only reasonable interpretation of the question.

2. Write only NOT_SURE if the example looks structurally valid, but you cannot reliably justify the provided answer.

3. Otherwise, write a concise explanation:
   - Identify the key fact, rule, or concept that makes the provided answer correct. Do not copy or paraphrase the whole question.
   - Explain briefly why the correct option follows from it. Avoid generic introductions.
   - Mention wrong options only when they are plausible or when contrasting them clarifies the answer.
   - Prefer 2-5 concise sentences unless the question genuinely requires more.
   - Do not mention that the answer was provided.
   - Do not express uncertainty about which option is correct.

4. End with the selected option letter inside \\boxed{{}}.
   - For example, if the correct answer is A, end with \\boxed{{A}}.
   - Do not add any text after the boxed answer.

Do not output anything except the concise explanation followed by the final \\boxed{{}} answer, unless Rule 1 or Rule 2 applies.
   - If Rule 1 applies, output exactly: FAIL.
   - If Rule 2 applies, output exactly: NOT_SURE.
"""


def _lang(ex: Dict) -> str:
    return LANGUAGE_NAMES.get(str(ex.get("language", "")).lower(), "the same language as the question")


def _system_message(language: str) -> str:
    return SYSTEM_MESSAGE.format(language=language)

def _system_message_fallback(language: str) -> str:
    return SYSTEM_MESSAGE_FALLBACK.format(language=language)


def _user_message(prompt: str, language: str, no_think: bool = False) -> str:
    return (
        f"The question is written in {language}. "
        f"Solve it independently. Reason and write your explanation in {language}. "
        f"If you are not confident, write only: NOT_SURE.\n\n"
        f"End with the final answer inside \\boxed{{}}, for example \\boxed{{A}}. \n\n"
        f"{prompt}\n\n"
    )
    

def _user_message_fallback(prompt: str, gold: str, language: str) -> str:
    return (
        f"The question is written in {language}. "
        f"If the question, options, or provided answer appear corrupted or inconsistent, write only: FAIL. "
        f"If the example looks valid but you cannot reliably justify the answer, write only: NOT_SURE.\n\n"
        f"Otherwise, end valid explanations with the final answer inside \\boxed{{}}, for example \\boxed{{A}}.\n\n"
        f"{prompt}\n\n"
        f"The correct answer is {gold}. Explain why, using {language}."
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
    temperature: float,
    top_p: float,
    retries: int = DEFAULT_MAX_RETRIES,
) -> Optional[str]:
    payload = {
        "model":       model,
        "messages":    messages,
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "top_p":       top_p,
        "stream":      False,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for attempt in range(retries):
        try:
            resp = await client.post(
                f"{API_BASE}/chat/completions",
                json=payload, headers=headers, timeout=180.0,
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


def _extract_think(text: str) -> str:
    m = re.search(r'<think>(.*?)</think>', text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _extract_boxed(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    for src in (_strip_think(text), text):
        matches = re.findall(r'\\boxed\{([^}]+)\}', src)
        if matches:
            return matches[-1].strip().upper()
    return None


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _too_short(text: str) -> bool:
    return (len(text) <= 20) if _contains_cjk(text) else (len(text.split()) <= 5)


def _too_long(text: str) -> bool:
    return (len(text) >= 1100) if _contains_cjk(text) else (len(text.split()) >= 220)

def _ends_with_expected_boxed(clean_stripped: str, expected: str) -> bool:
    tail = clean_stripped
    tail = re.sub(r"[。.!！\s]+$", "", tail)
    tail = re.sub(r"\s*\([^)]*\)\s*$", "", tail)  # strip trailing parenthetical
    return tail.endswith(expected)


def _validate_rationale(clean: Optional[str], boxed: Optional[str], gold: str) -> Optional[str]:
    """Return invalid_reason string, or None if valid."""
    expected = f"\\boxed{{{gold}}}"
    if not clean:
        return "empty"
    clean_stripped = clean.strip()
    clean_upper = clean_stripped.upper()
    if clean_upper == "FAIL":
        return "fail_corrupted_example"
    if "NOT_SURE" in clean_upper:
        return "not_sure"
    if boxed != gold:
        return f"wrong_answer:{boxed}"
    if _too_short(clean):
        return "too_short"
    if _too_long(clean):
        return "too_long"
    if not _ends_with_expected_boxed(clean_stripped, expected):
        return "not_ending_with_boxed"
    return None

# ---------------------------------------------------------------------------
# Per-example distillation — independent attempts + gold-answer fallback
# ---------------------------------------------------------------------------

def _attempt_record(
    stage: str,
    model: str,
    attempt: int,
    temperature: float,
    raw: Optional[str],
    clean: Optional[str],
    boxed: Optional[str],
    invalid_reason: Optional[str],
) -> Dict[str, Any]:
    return {
        "stage": stage,
        "model": model,
        "attempt": attempt,
        "temperature": temperature,
        "predicted": boxed,
        "valid": invalid_reason is None,
        "invalid_reason": invalid_reason,
        "raw": raw,
        "clean": clean,
    }


async def _run_teacher_attempt(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    api_key: str,
    model: str,
    messages: List[Dict],
    max_tokens: int,
    temperature: float,
    top_p: float,
    gold: str,
) -> Dict[str, Any]:
    async with sem:
        raw = await _call_api(
            client=client,
            model=model,
            messages=messages,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    if not raw:
        return {
            "raw": raw,
            "clean": None,
            "think_content": "",
            "boxed": None,
            "invalid_reason": "api_failure",
        }

    think_content = _extract_think(raw)
    clean = _strip_think(raw).strip()
    boxed = _extract_boxed(raw)
    invalid_reason = _validate_rationale(clean, boxed, gold)
    return {
        "raw": raw,
        "clean": clean,
        "think_content": think_content,
        "boxed": boxed,
        "invalid_reason": invalid_reason,
    }


async def _distill_example(
    ex: Dict,
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    api_key: str,
    sample_temperature: float,
    top_p: float,
) -> Dict:
    gold     = str(ex["answer"]).strip().upper()
    prompt   = ex["messages"][1]["content"]
    language = _lang(ex)

    raw            = None
    clean          = None
    think_content  = ""
    boxed          = None
    invalid_reason = "no_teacher_succeeded"
    used_teacher   = None
    valid          = False
    mode           = None
    attempts: List[Dict[str, Any]] = []

    llama_temperatures = [0.0] + [sample_temperature] * N_ADDITIONAL_RETRIES_MAIN
    apertus_temperatures = [0.0] + [sample_temperature] * N_ADDITIONAL_RETRIES_FALLBACK

    # 1) Llama independent attempts.
    for i, temperature in enumerate(llama_temperatures, start=1):
        messages = [
            {"role": "system", "content": _system_message(language)},
            {"role": "user",   "content": _user_message(prompt, language, USE_NO_THINK_PRIMARY)},
        ]
        out = await _run_teacher_attempt(
            client=client,
            sem=sem,
            api_key=api_key,
            model=TEACHER_PRIMARY,
            messages=messages,
            max_tokens=MAX_TOKENS_PRIMARY,
            temperature=temperature,
            top_p=top_p,
            gold=gold,
        )
        attempts.append(_attempt_record(
            "independent", TEACHER_PRIMARY, i, temperature,
            out["raw"], out["clean"], out["boxed"], out["invalid_reason"],
        ))
        raw, clean, think_content, boxed = out["raw"], out["clean"], out["think_content"], out["boxed"]
        invalid_reason = out["invalid_reason"]
        if invalid_reason is None:
            valid = True
            used_teacher = TEACHER_PRIMARY
            mode = "independent"
            break

    # 2) Apertus independent attempts.
    if not valid:
        for i, temperature in enumerate(apertus_temperatures, start=1):
            messages = [
                {"role": "system", "content": _system_message(language)},
                {"role": "user",   "content": _user_message(prompt, language, USE_NO_THINK_FALLBACK)},
            ]
            out = await _run_teacher_attempt(
                client=client,
                sem=sem,
                api_key=api_key,
                model=TEACHER_FALLBACK,
                messages=messages,
                max_tokens=MAX_TOKENS_FALLBACK,
                temperature=temperature,
                top_p=top_p,
                gold=gold,
            )
            attempts.append(_attempt_record(
                "independent", TEACHER_FALLBACK, i, temperature,
                out["raw"], out["clean"], out["boxed"], out["invalid_reason"],
            ))
            raw, clean, think_content, boxed = out["raw"], out["clean"], out["think_content"], out["boxed"]
            invalid_reason = out["invalid_reason"]
            if invalid_reason is None:
                valid = True
                used_teacher = TEACHER_FALLBACK
                mode = "independent"
                break

    # # 3) Gold-answer fallback with Llama, then Apertus.
    # if not valid:
    #     fallback_teachers = [
    #         (TEACHER_PRIMARY, MAX_TOKENS_PRIMARY),
    #     ]
    #     for i, (model, max_tokens) in enumerate(fallback_teachers, start=1):
    #         messages = [
    #             {"role": "system", "content": _system_message_fallback(language)},
    #             {"role": "user",   "content": _user_message_fallback(prompt, gold, language)},
    #         ]
    #         out = await _run_teacher_attempt(
    #             client=client,
    #             sem=sem,
    #             api_key=api_key,
    #             model=model,
    #             messages=messages,
    #             max_tokens=max_tokens,
    #             temperature=0.0,
    #             top_p=top_p,
    #             gold=gold,
    #         )
    #         attempts.append(_attempt_record(
    #             "gold_fallback", model, i, 0.0,
    #             out["raw"], out["clean"], out["boxed"], out["invalid_reason"],
    #         ))
    #         raw, clean, think_content, boxed = out["raw"], out["clean"], out["think_content"], out["boxed"]
    #         invalid_reason = out["invalid_reason"]
    #         if invalid_reason == "fail_corrupted_example":
    #             valid = False
    #             used_teacher = model
    #             mode = "gold_fallback_fail"
    #             break
    #         if invalid_reason == "not_sure":
    #             valid = False
    #             used_teacher = model
    #             mode = "gold_fallback_not_sure"
    #             break
    #         if invalid_reason is None:
    #             valid = True
    #             used_teacher = model
    #             mode = "gold_fallback"
    #             break
    # 3) Gold-answer fallback with Llama — skip if there was genuine uncertainty.
    if not valid:
        last_apertus_attempt = next(
            (a for a in reversed(attempts) if a.get("model") == TEACHER_FALLBACK),
            None,
        )
        any_llama_not_sure = any(
            a.get("model") == TEACHER_PRIMARY and a.get("invalid_reason") == "not_sure"
            for a in attempts
        )
        apertus_was_wrong_not_sure = (
            last_apertus_attempt is not None and
            last_apertus_attempt.get("invalid_reason") != "not_sure"
        )
        skip_gold_fallback = (
            # Case 1: Apertus itself said NOT_SURE
            (last_apertus_attempt is not None and
             last_apertus_attempt.get("invalid_reason") == "not_sure")
            or
            # Case 2: Llama was NOT_SURE at some point AND Apertus just got wrong answer
            (any_llama_not_sure and apertus_was_wrong_not_sure)
        )

        if skip_gold_fallback:
            invalid_reason = "skipped_gold_fallback_uncertainty"
            mode           = "discarded_not_sure"
        else:
            fallback_teachers = [
                (TEACHER_PRIMARY, MAX_TOKENS_PRIMARY),
            ]
            for i, (model, max_tokens) in enumerate(fallback_teachers, start=1):
                messages = [
                    {"role": "system", "content": _system_message_fallback(language)},
                    {"role": "user",   "content": _user_message_fallback(prompt, gold, language)},
                ]
                out = await _run_teacher_attempt(
                    client=client,
                    sem=sem,
                    api_key=api_key,
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    top_p=top_p,
                    gold=gold,
                )
                attempts.append(_attempt_record(
                    "gold_fallback", model, i, 0.0,
                    out["raw"], out["clean"], out["boxed"], out["invalid_reason"],
                ))
                raw, clean, think_content, boxed = out["raw"], out["clean"], out["think_content"], out["boxed"]
                invalid_reason = out["invalid_reason"]
                if invalid_reason == "fail_corrupted_example":
                    valid = False
                    used_teacher = model
                    mode = "gold_fallback_fail"
                    break
                if invalid_reason == "not_sure":
                    valid = False
                    used_teacher = model
                    mode = "gold_fallback_not_sure"
                    break
                if invalid_reason is None:
                    valid = True
                    used_teacher = model
                    mode = "gold_fallback"
                    break

    result = {
        **ex,
        "distill": {
            "gold":           gold,
            "predicted":      boxed,
            "correct":        (boxed == gold),
            "valid":          valid,
            "invalid_reason": invalid_reason if not valid else None,
            "used_teacher":   used_teacher,
            "mode":           mode,
            "raw":            raw,
            "think_content":  think_content,
            "clean":          clean,
            "attempts":       attempts,
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
# Progress + I/O helpers  (unchanged)
# ---------------------------------------------------------------------------

class _Progress:
    def __init__(self, total: int, log_every: int):
        self.total, self.log_every = total, log_every
        self.done = self.valid = 0
        self.t0 = time.time()

    def update(self, valid: bool) -> None:
        self.done += 1
        if valid:
            self.valid += 1
        if self.done % self.log_every == 0 or self.done == self.total:
            elapsed = time.time() - self.t0
            rate    = self.done / elapsed if elapsed > 0 else 0
            eta     = (self.total - self.done) / rate if rate > 0 else 0
            pct     = 100 * self.valid // max(1, self.done)
            print(f"    {self.done}/{self.total} "
                  f"| valid={self.valid} ({pct}%) "
                  f"| {rate:.1f} ex/s | ETA {eta:.0f}s")


def _append_jsonl(records: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _load_done_ids(path: Path) -> tuple[set, List[Dict]]:
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
# Record normalization  (unchanged)
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
# File-level pipeline  (unchanged except progress update field)
# ---------------------------------------------------------------------------

async def _distill_file(
    src_path: Path,
    all_path: Path,
    valid_path: Path,
    fail_path: Path,
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    api_key: str,
    dry_run: bool,
    log_every: int,
    save_every: int,
    debug_n: int,
    max_valid_per_file: Optional[int],
    shuffle_seed: int,
    sample_temperature: float,
    top_p: float,
) -> Dict[str, Any]:
    records = _load_processed(src_path)
    records = [r for r in records if r.get("task_type") == "mcq"]

    rng = random.Random(shuffle_seed)
    rng.shuffle(records)

    if dry_run:
        records = records[:min(debug_n, len(records))]

    if not records:
        return {"total": 0, "valid": 0}

    # seed path
    seed_path = all_path.parent / f"{all_path.stem}_shuffle_meta.json"
    if not seed_path.exists():
        with open(seed_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "source_file": str(src_path),
                    "shuffle_seed": shuffle_seed,
                    "max_valid_per_file": max_valid_per_file,
                    "n_records_after_filtering": len(records),
                    "dry_run": dry_run,
                    "debug_n": debug_n,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    done_ids, existing = _load_done_ids(all_path)
    if done_ids:
        print(f"  Resuming: {len(done_ids)} already done")

    # existing valid ids
    existing_valid = sum(1 for r in existing if r["distill"]["valid"])
    if max_valid_per_file is not None and existing_valid >= max_valid_per_file:
        print(f"  Valid cap already reached: {existing_valid}/{max_valid_per_file}")
        return {"total": len(existing), "valid": existing_valid, "stopped_reason": "valid_cap_reached"}
        
    todo = [r for r in records if r["source_id"] not in done_ids]
    print(f"  Processing: {len(todo)} remaining / {len(records)} total")

    if not todo:
        valid_count = sum(1 for r in existing if r["distill"]["valid"])
        print(f"  Already complete. Valid: {valid_count}/{len(existing)}")
        return {"total": len(records), "valid": valid_count}

    prog = _Progress(len(todo), log_every)

    async def _run(ex):
        result = await _distill_example(
            ex=ex,
            client=client,
            sem=sem,
            api_key=api_key,
            sample_temperature=sample_temperature,
            top_p=top_p,
        )
        prog.update(valid=result["distill"]["valid"])
        return result

    # batch loop
    valid_so_far = existing_valid
    stopped_reason = "exhausted"
    for batch_start in range(0, len(todo), save_every):
        if max_valid_per_file is not None and valid_so_far >= max_valid_per_file:
            stopped_reason = "valid_cap_reached"
            break
        batch = todo[batch_start:batch_start + save_every]
        results = await asyncio.gather(*[_run(ex) for ex in batch])
        if max_valid_per_file is not None:
            kept_results = []
            valid_batch = []
            for r in results:
                if r["distill"]["valid"] and valid_so_far >= max_valid_per_file:
                    stopped_reason = "valid_cap_reached"
                    break
                kept_results.append(r)
                if r["distill"]["valid"]:
                    valid_batch.append(r)
                    valid_so_far += 1
            results = kept_results
        else:
            valid_batch = [r for r in results if r["distill"]["valid"]]
            valid_so_far += len(valid_batch)

        fail_batch = [
            r for r in results
            if r["distill"].get("invalid_reason") == "fail_corrupted_example"
            or r["distill"].get("mode") == "discarded_not_sure"
            or any(
                a.get("invalid_reason") == "fail_corrupted_example"
                for a in r["distill"].get("attempts", [])
            )
        ]

        _append_jsonl(results, all_path)
        if valid_batch:
            _append_jsonl(valid_batch, valid_path)
        if fail_batch:
            _append_jsonl(fail_batch, fail_path)
        if max_valid_per_file is not None and valid_so_far >= max_valid_per_file:
            stopped_reason = "valid_cap_reached"
            break

    _, all_results = _load_done_ids(all_path)
    valid_total    = sum(1 for r in all_results if r["distill"]["valid"])
    fail_total     = sum(
        1 for r in all_results
        if r["distill"].get("invalid_reason") == "fail_corrupted_example"
    )
    invalid_reasons: Dict[str, int] = {}
    for r in all_results:
        reason = r["distill"].get("invalid_reason")
        if reason:
            invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1

    teacher_counts: Dict[str, int] = {}
    mode_counts: Dict[str, int] = {}
    for r in all_results:
        t = r["distill"].get("used_teacher") or "none"
        m = r["distill"].get("mode") or "none"
        teacher_counts[t] = teacher_counts.get(t, 0) + 1
        mode_counts[m] = mode_counts.get(m, 0) + 1

    pct = 100 * valid_total // max(1, len(all_results))
    print(f"  Valid: {valid_total}/{len(all_results)} ({pct}%)")
    print(f"  FAIL cases: {fail_total}")
    print(f"  Teachers used: {teacher_counts}")
    print(f"  Modes used: {mode_counts}")
    if invalid_reasons:
        print(f"  Invalid reasons: {invalid_reasons}")

    return {
        "total": len(all_results),
        "valid": valid_total,
        "fail": fail_total,
        "invalid_reasons": invalid_reasons,
        "teacher_counts": teacher_counts,
        "mode_counts": mode_counts,
        "stopped_reason": stopped_reason,
        "max_valid_per_file": max_valid_per_file,
        "shuffle_seed": shuffle_seed,
    }

# ---------------------------------------------------------------------------
# Main  (unchanged)
# ---------------------------------------------------------------------------

SUPPORTED_DATASETS = list(PROCESSED_INPUTS.keys())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets",      nargs="+", default=SUPPORTED_DATASETS, choices=SUPPORTED_DATASETS)
    p.add_argument("--processed_dir", type=Path, default=Path("multilingual/data/processed"))
    p.add_argument("--output_dir",    type=Path, default=Path("multilingual/data/distilled/v1stage"))
    p.add_argument("--concurrency",   type=int,  default=DEFAULT_CONCURRENCY)
    p.add_argument("--log_every",     type=int,  default=LOG_EVERY)
    p.add_argument("--save_every",    type=int,  default=SAVE_EVERY)
    p.add_argument("--dry_run",       action="store_true")
    p.add_argument("--debug_n",       type=int,  default=5)
    p.add_argument("--max_valid_per_file", type=int, default=None)
    p.add_argument("--shuffle_seed", type=int, default=42)
    p.add_argument("--sample_temperature", type=float, default=DEFAULT_SAMPLE_TEMPERATURE)
    p.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    return p.parse_args()


def _build_jobs(processed_dir: Path, datasets: List[str]) -> List[tuple]:
    jobs = []
    for dataset in datasets:
        for rel_path, base in PROCESSED_INPUTS[dataset]:
            path = processed_dir / rel_path
            if not path.exists():
                raise FileNotFoundError(f"Processed file not found: {path}")
            jobs.append((path, base))
    return jobs


async def main_async(args):
    api_key = os.environ.get("CSCS_SERVING_API", "")
    if not api_key:
        raise ValueError("CSCS_SERVING_API not set.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = _build_jobs(args.processed_dir, args.datasets)

    print(f"Primary teacher:  {TEACHER_PRIMARY}")
    print(f"Fallback teacher: {TEACHER_FALLBACK}")
    print(f"Sampling: first independent attempt temperature=0.0; later attempts temperature={args.sample_temperature}, top_p={args.top_p}")
    print(f"Jobs ({len(jobs)}):")
    for path, base in jobs:
        print(f"  {path.name} → {base}")
    if args.dry_run:
        print(f"\n[DRY RUN] {args.debug_n} examples per file.\n")

    sem     = asyncio.Semaphore(args.concurrency)
    summary = []

    async with httpx.AsyncClient() as client:
        for src_path, base in jobs:
            all_path   = args.output_dir / f"{base}_all.jsonl"
            valid_path = args.output_dir / f"{base}_distilled.jsonl"
            fail_path  = args.output_dir / f"{base}_fail.jsonl"

            print(f"\n── {src_path.name} ──")
            t0    = time.time()
            stats = await _distill_file(
                src_path, all_path, valid_path, fail_path,
                client, sem, api_key,
                args.dry_run, args.log_every, args.save_every, args.debug_n,
                args.max_valid_per_file, args.shuffle_seed,
                args.sample_temperature, args.top_p,
            )
            elapsed          = time.time() - t0
            stats["file"]    = src_path.name
            stats["elapsed"] = round(elapsed, 1)
            summary.append(stats)
            print(f"  Done in {elapsed:.1f}s")

    print("\n" + "="*60)
    print("DISTILLATION SUMMARY")
    print("="*60)
    tot, tot_v = 0, 0
    for s in summary:
        pct = 100 * s["valid"] // max(1, s["total"])
        print(f"  {s['file']:50s}  total={s['total']:4d}  valid={s['valid']:4d} ({pct}%)  stop={s.get('stopped_reason', 'unknown')}  ({s['elapsed']}s)")
        tot   += s["total"]
        tot_v += s["valid"]
    pct_total = 100 * tot_v // max(1, tot)
    print(f"\n  TOTAL: {tot} → valid {tot_v} ({pct_total}%)")

    summary_path = args.output_dir / "distill_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary: {summary_path}")


def main():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()