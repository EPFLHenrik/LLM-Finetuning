"""code/multilinguality_model/data_pipeline/generation/generate_italian_civil_code_mcq.py

Generate verified Italian Civil Code MCQs using teacher models.
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
DEFAULT_GEN_TEMPERATURE = 0.45   # diverse question styles
VER_TEMPERATURE         = 0.0   # deterministic verification
DEFAULT_TOP_P           = 0.95
LOG_EVERY               = 25
SAVE_EVERY              = 25
MIN_ARTICLE_CHARS       = 80    # skip very short articles

DATASET_NAME = "AndreaSimeri/Italian_Civil_Code"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_GEN = """Sei un esperto di diritto civile italiano e di costruzione di quesiti
giuridici a scelta multipla. Devi generare UNA sola domanda MCQ in
italiano a partire dall'articolo del Codice Civile fornito.

KEPT solo se:
1. La domanda e' rispondibile direttamente dal testo dell'articolo.
2. Testa una condizione, soggetto, effetto giuridico, termine,
   competenza, eccezione o conseguenza prevista dall'articolo.
3. Formulazione giuridicamente naturale, formale e grammaticale.

Criteri per i distrattori:
1. Plausibili per uno studente di diritto, ma chiaramente errati.
2. Stessa categoria della risposta corretta (soggetti con soggetti,
   termini con termini, ecc.).
3. Modificano un elemento giuridicamente rilevante.
4. Simili per lunghezza, stile e specificita'.

Se l'articolo e' troppo breve, incompleto, corrotto o puramente
abrogativo, rispondi SKIP.

Formato output (JSON puro, senza testo aggiuntivo):
{
  "question": "...",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "answer": "A"
}

Se l'articolo non consente una buona domanda, rispondi esattamente:
SKIP"""

SYSTEM_VER = """Sei un verificatore severo di domande MCQ di diritto civile italiano.

Ti verranno forniti:
1. Il testo originale dell'articolo del Codice Civile.
2. Una domanda MCQ con quattro opzioni A, B, C, D.
3. La risposta corretta indicata.

Rispondi VALID solo se TUTTI i criteri sono soddisfatti:
1. La domanda è direttamente fondata sull'articolo.
2. La risposta corretta è chiaramente e direttamente supportata dal testo.
3. Esiste una sola risposta corretta.
4. Nessun distrattore è parzialmente corretto.
5. I distrattori sono plausibili ma giuridicamente errati rispetto all'articolo.
6. I distrattori modificano elementi giuridicamente rilevanti, non dettagli casuali.
7. Le opzioni appartengono alla stessa categoria logica e giuridica.
8. Le opzioni sono comparabili per lunghezza, stile e specificità.
9. La risposta corretta non è resa ovvia perché è molto più lunga, più precisa o più vicina al testo dell'articolo.
10. La domanda non è eccessivamente banale se l'articolo consentiva una domanda migliore.
11. La domanda non si basa su testo abrogato, corrotto, incompleto o su note redazionali confuse.
12. La lingua italiana è corretta e il registro è giuridico-formale.

Rifiuta subito se:
- la risposta corretta non è supportata;
- più opzioni sono corrette o parzialmente corrette;
- i distrattori sono solo assurdi o inventati;
- i distrattori sono solo non menzionati, ma non incompatibili con l'articolo;
- la risposta corretta è l'unica opzione completa;
- la domanda copia meccanicamente l'articolo senza costruire un vero quesito;
- l'articolo è troppo breve o rumoroso per una buona domanda;
- il quesito richiede conoscenza esterna.

Output ammessi, esattamente uno:
- VALID
- INVALID:risposta_non_supportata
- INVALID:piu_risposte_corrette
- INVALID:distrattori_parzialmente_corretti
- INVALID:distrattori_deboli
- INVALID:opzioni_sbilanciate
- INVALID:troppo_banale
- INVALID:domanda_troppo_meccanica
- INVALID:testo_articolo_inadeguato
- INVALID:richiede_conoscenza_esterna
- INVALID:italiano_non_naturale
- NOT_SURE

Non aggiungere nessun altro testo."""


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
    p.add_argument("--output_dir",      type=Path,  default=Path("data/multilinguality/processed/italian_civil_code"))
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