"""multilingual/data_pipeline/generation/generate_italian_civil_code_mcq.py

Generates Italian MCQ questions from the Italian Civil Code dataset
(AndreaSimeri/Italian_Civil_Code) using a two-stage pipeline:

Stage 1 — Generation (Llama):
  Read an article and generate one 4-option MCQ with a clear correct answer
  and three plausible but clearly wrong distractors.

Stage 2 — Verification (Llama primary, Apertus secondary):
  Given the article text, the generated MCQ, and the gold answer, verify:
    - The correct answer is directly supported by the article.
    - The three wrong answers are clearly incorrect (contradict the article,
      not merely complementary or unrelated).
  Output: VALID, INVALID:<reason>, or NOT_SURE.

Saving is append-only and resumable:
    <output_dir>/italian_civil_code_all.jsonl       ← all attempts
    <output_dir>/italian_civil_code_generated.jsonl ← verified valid MCQs
    <output_dir>/italian_civil_code_fail.jsonl      ← rejected examples

Usage:
    export CSCS_SERVING_API="sk-rc-..."

    python -m multilingual.data_pipeline.generation.generate_italian_civil_code_mcq \\
        --output_dir multilingual/data/processed/italian_civil_code \\
        --concurrency 20 \\
        --max_valid 2000 \\
        --dry_run
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
import random
import httpx

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE          = "https://api.swissai.svc.cscs.ch/v1"
TEACHER_GEN       = "meta-llama/Llama-3.3-70B-Instruct"
TEACHER_VER_1     = "meta-llama/Llama-3.3-70B-Instruct"
TEACHER_VER_2     = "swiss-ai/Apertus-70B-Instruct-2509"

MAX_TOKENS_GEN    = 600
MAX_TOKENS_VER    = 400

DEFAULT_CONCURRENCY     = 8
DEFAULT_MAX_RETRIES     = 3
DEFAULT_RETRY_DELAY     = 1.0
DEFAULT_GEN_TEMPERATURE = 0.7   # diverse question styles
VER_TEMPERATURE         = 0.0   # deterministic verification
DEFAULT_TOP_P           = 0.95
LOG_EVERY               = 25
SAVE_EVERY              = 25
MIN_ARTICLE_CHARS       = 80    # skip very short articles

DATASET_NAME = "AndreaSimeri/Italian_Civil_Code"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_GEN = """Sei un esperto di diritto italiano. Il tuo compito è generare una domanda a scelta multipla (MCQ) di alta qualità in italiano a partire da un articolo del Codice Civile italiano.

Regole per la domanda:
1. La domanda deve essere rispondibile direttamente dal testo dell'articolo fornito.
2. La risposta corretta deve essere letteralmente supportata dall'articolo — nessuna parafrasi che cambia il significato.
3. Le tre risposte sbagliate (distrattori) devono essere plausibili ma chiaramente errate:
   - Modifica un dettaglio chiave della risposta corretta (ad es. cambia un soggetto, un termine legale, un limite temporale).
   - Oppure combina concetti reali in modo errato.
   - Oppure usa la negazione della risposta corretta.
   - I distrattori devono CONTRADDIRE l'articolo, non fornire informazioni complementari.
4. Tutte le opzioni devono essere simili in lunghezza e stile. Se la risposta corretta è lunga, anche i distrattori devono essere di lunghezza comparabile. Non usare opzioni molto corte accanto a opzioni molto lunghe.
5. Una sola risposta deve essere corretta.
6. La domanda deve essere in italiano formale e registro giuridico.

Formato di output (JSON puro, nessun testo aggiuntivo):
{
  "question": "...",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "answer": "A"
}

Se l'articolo è troppo corto, incompleto o non contiene informazioni sufficienti per generare una buona domanda, rispondi esattamente: SKIP"""


SYSTEM_VER = """Sei un esperto di diritto italiano incaricato di verificare la qualità di domande MCQ generate dal Codice Civile italiano.

Ti verrà fornito:
1. Il testo originale dell'articolo (fonte di verità).
2. Una domanda MCQ con 4 opzioni (A, B, C, D) e la risposta corretta indicata.

Il tuo compito è verificare:
1. La risposta corretta è direttamente e chiaramente supportata dall'articolo? (non inferita, non parafrasata in modo distorto)
2. Ciascuna delle tre risposte sbagliate CONTRADDICE l'articolo? (non sono semplicemente informazioni non presenti — devono essere attivamente errate rispetto al testo)
3. C'è una sola risposta inequivocabilmente corretta?
4. La domanda è grammaticalmente corretta in italiano?

Output possibili:
- Rispondi esattamente VALID se tutti i criteri sono soddisfatti.
- Rispondi esattamente INVALID:<motivo breve> se uno o più criteri non sono soddisfatti.
  Esempi: INVALID:risposta_corretta_non_supportata, INVALID:distrattori_non_contraddicono, INVALID:piu_risposte_corrette
- Rispondi esattamente NOT_SURE se non riesci a determinare la correttezza con certezza.

Non aggiungere nessun altro testo oltre al tuo output."""


def _gen_user_message(article_id: str, title: str, text: str) -> str:
    return (
        f"Articolo: {article_id} — {title}\n\n"
        f"Testo:\n{text}\n\n"
        f"Genera una domanda MCQ a partire da questo articolo."
    )


def _ver_user_message(
    article_id: str, title: str, text: str,
    question: str, options: Dict[str, str], answer: str,
) -> str:
    opts = "\n".join(f"{k}) {v}" for k, v in options.items())
    return (
        f"Articolo: {article_id} — {title}\n\n"
        f"Testo originale:\n{text}\n\n"
        f"---\n"
        f"Domanda MCQ generata:\n{question}\n\n"
        f"{opts}\n\n"
        f"Risposta corretta indicata: {answer}\n\n"
        f"Verifica la qualità di questa MCQ secondo i criteri forniti."
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
# Parsing helpers
# ---------------------------------------------------------------------------

def _shuffle_options(mcq: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    """Randomly permute options and remap answer letter."""
    letters = ["A", "B", "C", "D"]
    values  = [mcq["options"][l] for l in letters]
    correct = mcq["options"][mcq["answer"]]
    rng.shuffle(values)
    new_options = dict(zip(letters, values))
    new_answer  = next(l for l, v in new_options.items() if v == correct)
    return {**mcq, "options": new_options, "answer": new_answer}


def _parse_mcq(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    """Parse JSON MCQ from generation output. Returns None on failure."""
    if not raw:
        return None
    raw_stripped = raw.strip()
    if raw_stripped.upper() == "SKIP":
        return None
    clean = re.sub(r"^```(?:json)?\s*", "", raw_stripped, flags=re.IGNORECASE)
    clean = re.sub(r"\s*```$", "", clean)
    try:
        parsed = json.loads(clean.strip())
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', clean, re.DOTALL)
        if not m:
            return None
        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None

    if not isinstance(parsed.get("question"), str):
        return None
    opts = parsed.get("options", {})
    if not all(k in opts for k in ("A", "B", "C", "D")):
        return None
    answer = str(parsed.get("answer", "")).strip().upper()
    if answer not in ("A", "B", "C", "D"):
        return None

    return {
        "question": parsed["question"].strip(),
        "options":  {k: str(v).strip() for k, v in opts.items()},
        "answer":   answer,
    }


def _parse_verification(raw: Optional[str]) -> str:
    """Return 'VALID', 'INVALID:<reason>', 'NOT_SURE', or 'api_failure'."""
    if not raw:
        return "api_failure"
    r = raw.strip().upper()
    if r == "VALID":
        return "VALID"
    if r == "NOT_SURE":
        return "NOT_SURE"
    if r.startswith("INVALID"):
        return raw.strip()  # preserve original case for reason
    return f"unexpected_output:{raw[:80]}"

# ---------------------------------------------------------------------------
# Per-article pipeline
# ---------------------------------------------------------------------------

async def _process_article(
    article: Dict[str, Any],
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    api_key: str,
    gen_temperature: float,
    top_p: float,
) -> Dict[str, Any]:
    article_id = str(article.get("article_id", ""))
    title      = str(article.get("article_title", "")).strip()
    text       = str(article.get("article_text",  "")).strip()

    result = {
        "source_id":   f"italian_civil_code_{article_id}",
        "dataset":     "italian_civil_code",
        "language":    "italian",
        "task_type":   "mcq",
        "article_id":  article_id,
        "article_title": title,
        "article_text":  text,
        "mcq":         None,
        "generation":  {"raw": None, "parsed": None, "valid": False, "reason": None},
        "verification": {"ver1": None, "ver2": None, "final": None},
        "valid":       False,
        "prompt":      None,
        "answer":      None,
        "metadata":    {
            "article_id":    article_id,
            "article_title": title,
            "generation_model":    TEACHER_GEN,
            "verification_model1": TEACHER_VER_1,
            "verification_model2": TEACHER_VER_2,
        },
    }

    # ── Stage 1: Generation ──────────────────────────────────────────────────
    gen_messages = [
        {"role": "system", "content": SYSTEM_GEN},
        {"role": "user",   "content": _gen_user_message(article_id, title, text)},
    ]
    async with sem:
        gen_raw = await _call_api(
            client=client, model=TEACHER_GEN, messages=gen_messages,
            api_key=api_key, max_tokens=MAX_TOKENS_GEN,
            temperature=gen_temperature, top_p=top_p,
        )

    result["generation"]["raw"] = gen_raw

    if not gen_raw or gen_raw.strip().upper() == "SKIP":
        result["generation"]["reason"] = "skipped_by_generator"
        return result

    mcq = _parse_mcq(gen_raw)
    if mcq is None:
        result["generation"]["reason"] = "parse_failed"
        return result

    # shuffle options to avoid positional bias in verification, using a seed for reproducibility
    rng = random.Random(article_id)
    mcq = _shuffle_options(mcq, rng)

    result["generation"]["parsed"] = mcq
    result["generation"]["valid"]  = True
    result["mcq"] = mcq

    # ── Stage 2a: Verification — Llama ──────────────────────────────────────
    ver_messages = [
        {"role": "system", "content": SYSTEM_VER},
        {"role": "user",   "content": _ver_user_message(
            article_id, title, text,
            mcq["question"], mcq["options"], mcq["answer"],
        )},
    ]
    async with sem:
        ver1_raw = await _call_api(
            client=client, model=TEACHER_VER_1, messages=ver_messages,
            api_key=api_key, max_tokens=MAX_TOKENS_VER,
            temperature=VER_TEMPERATURE, top_p=top_p,
        )

    ver1_result = _parse_verification(ver1_raw)
    result["verification"]["ver1"] = {"raw": ver1_raw, "result": ver1_result}

    if ver1_result == "VALID":
        # Passed primary verification — accepted
        result["verification"]["final"] = "VALID"
        result["valid"] = True
    elif ver1_result in ("NOT_SURE", "api_failure"):
        # ── Stage 2b: Verification — Apertus (secondary) ────────────────────
        async with sem:
            ver2_raw = await _call_api(
                client=client, model=TEACHER_VER_2, messages=ver_messages,
                api_key=api_key, max_tokens=MAX_TOKENS_VER,
                temperature=VER_TEMPERATURE, top_p=top_p,
            )
        ver2_result = _parse_verification(ver2_raw)
        result["verification"]["ver2"] = {"raw": ver2_raw, "result": ver2_result}
        result["verification"]["final"] = ver2_result

        if ver2_result == "VALID":
            result["valid"] = True
        else:
            result["generation"]["reason"] = f"verification_failed:{ver2_result}"
    else:
        # Primary verification returned INVALID
        result["verification"]["final"] = ver1_result
        result["generation"]["reason"]  = f"verification_failed:{ver1_result}"

    # ── Build final SFT fields if valid ─────────────────────────────────────
    if result["valid"] and mcq:
        opts_str = "\n".join(f"{k}) {v}" for k, v in mcq["options"].items())
        result["prompt"] = f"{mcq['question']}\n\n{opts_str}"
        result["answer"] = mcq["answer"]

    return result

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

class _Progress:
    def __init__(self, total: int, log_every: int):
        self.total, self.log_every = total, log_every
        self.done = self.valid = 0
        self.t0   = time.time()

    def update(self, valid: bool) -> None:
        self.done += 1
        if valid:
            self.valid += 1
        if self.done % self.log_every == 0 or self.done == self.total:
            elapsed = time.time() - self.t0
            rate    = self.done / elapsed if elapsed > 0 else 0
            eta     = (self.total - self.done) / rate if rate > 0 else 0
            pct     = 100 * self.valid // max(1, self.done)
            print(f"    {self.done}/{self.total} | valid={self.valid} ({pct}%) "
                  f"| {rate:.1f} ex/s | ETA {eta:.0f}s")


def _append_jsonl(records: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _load_done_ids(path: Path) -> set:
    done = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                        done.add(r["source_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    return done


def _load_dataset() -> List[Dict[str, Any]]:
    """Load Italian Civil Code from HuggingFace."""
    from datasets import load_dataset
    print(f"Loading {DATASET_NAME} from HuggingFace...")
    ds = load_dataset(DATASET_NAME, split="train")
    articles = []
    for row in ds:
        text = str(row.get("article_text", "")).strip()
        if len(text) < MIN_ARTICLE_CHARS:
            continue
        articles.append(dict(row))
    print(f"Loaded {len(articles)} articles (filtered from {len(ds)})")
    return articles

# ---------------------------------------------------------------------------
# Main async pipeline
# ---------------------------------------------------------------------------

async def main_async(args):
    api_key = os.environ.get("CSCS_SERVING_API", "")
    if not api_key:
        raise ValueError("CSCS_SERVING_API not set.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_path   = args.output_dir / "italian_civil_code_all.jsonl"
    valid_path = args.output_dir / "italian_civil_code_generated.jsonl"
    fail_path  = args.output_dir / "italian_civil_code_fail.jsonl"

    articles = _load_dataset()
    if args.dry_run:
        articles = articles[:args.debug_n]

    done_ids = _load_done_ids(all_path)
    if done_ids:
        print(f"Resuming: {len(done_ids)} already processed")

    existing_valid = 0
    if all_path.exists():
        with open(all_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                        if r.get("valid"):
                            existing_valid += 1
                    except json.JSONDecodeError:
                        pass

    if args.max_valid and existing_valid >= args.max_valid:
        print(f"Valid cap already reached: {existing_valid}/{args.max_valid}")
        return

    todo = [a for a in articles
            if f"italian_civil_code_{a.get('article_id', '')}" not in done_ids]
    print(f"Processing: {len(todo)} remaining / {len(articles)} total")

    sem  = asyncio.Semaphore(args.concurrency)
    prog = _Progress(len(todo), LOG_EVERY)

    async def _run(article):
        r = await _process_article(
            article=article,
            client=client,
            sem=sem,
            api_key=api_key,
            gen_temperature=args.gen_temperature,
            top_p=args.top_p,
        )
        prog.update(valid=r["valid"])
        return r

    valid_so_far = existing_valid
    async with httpx.AsyncClient() as client:
        for batch_start in range(0, len(todo), SAVE_EVERY):
            if args.max_valid and valid_so_far >= args.max_valid:
                print(f"Valid cap reached: {valid_so_far}/{args.max_valid}")
                break

            batch   = todo[batch_start:batch_start + SAVE_EVERY]
            results = await asyncio.gather(*[_run(a) for a in batch])

            valid_batch = [r for r in results if r["valid"]]
            fail_batch  = [r for r in results if not r["valid"]]

            if args.max_valid:
                kept = []
                for r in valid_batch:
                    if valid_so_far >= args.max_valid:
                        break
                    kept.append(r)
                    valid_so_far += 1
                valid_batch = kept
            else:
                valid_so_far += len(valid_batch)

            _append_jsonl(results,     all_path)
            _append_jsonl(valid_batch, valid_path)
            _append_jsonl(fail_batch,  fail_path)

    # Summary
    total_processed = len(done_ids) + len(todo)
    pct = 100 * valid_so_far // max(1, total_processed)
    print(f"\nDone. Valid: {valid_so_far}/{total_processed} ({pct}%)")
    print(f"  Generated: {valid_path}")
    print(f"  All:       {all_path}")
    print(f"  Fail:      {fail_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir",      type=Path,  default=Path("multilingual/data/processed/italian_civil_code"))
    p.add_argument("--concurrency",     type=int,   default=DEFAULT_CONCURRENCY)
    p.add_argument("--gen_temperature", type=float, default=DEFAULT_GEN_TEMPERATURE)
    p.add_argument("--top_p",           type=float, default=DEFAULT_TOP_P)
    p.add_argument("--max_valid",       type=int,   default=None, help="Stop after N valid examples.")
    p.add_argument("--dry_run",         action="store_true")
    p.add_argument("--debug_n",         type=int,   default=5)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Generation model:      {TEACHER_GEN}")
    print(f"Verification model 1:  {TEACHER_VER_1}")
    print(f"Verification model 2:  {TEACHER_VER_2}")
    print(f"Min article chars:     {MIN_ARTICLE_CHARS}")
    print(f"Gen temperature:       {args.gen_temperature}")
    print(f"Concurrency:           {args.concurrency}")
    print(f"Output:                {args.output_dir}")
    if args.dry_run:
        print(f"[DRY RUN] {args.debug_n} articles\n")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()