"""code/multilinguality_model/data_pipeline/generation/filter_italian_mcq.py
Filter generated Italian MCQ candidates using teacher-model validation.
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

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE      = "https://api.swissai.svc.cscs.ch/v1"
TEACHER_PRI   = "meta-llama/Llama-3.3-70B-Instruct"
TEACHER_SEC   = "swiss-ai/Apertus-70B-Instruct-2509"

MAX_TOKENS_PRI = 200
MAX_TOKENS_SEC = 200

DEFAULT_CONCURRENCY  = 16
DEFAULT_MAX_RETRIES  = 3
DEFAULT_RETRY_DELAY  = 1.0
LOG_EVERY            = 25
SAVE_EVERY           = 50

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_FILTER = """\
Sei un classificatore severo di domande a scelta multipla in italiano.

Il tuo compito è decidere se una domanda MCQ deve essere KEPT o REJECTED
per un dataset di valutazione di conoscenza italiana e culturale.

Tieni la domanda (KEPT) solo se soddisfa TUTTI i seguenti criteri:
1. È scritta interamente in italiano corretto e comprensibile.
2. Riguarda la cultura, la storia, la letteratura, la lingua, le istituzioni,
   il diritto, la geografia, i personaggi, l'arte o la musica ITALIANI.
   Esempi validi: storia italiana, letteratura italiana (Dante, Manzoni, Verga, 
   Leopardi, Pirandello...), grammatica italiana, diritto italiano, 
   istituzioni italiane, geografia italiana, storia del Risorgimento, 
   Resistenza italiana, personaggi storici italiani, musica italiana.
3. Non è corrotta da errori OCR evidenti (lettere casuali inserite nel mezzo
   delle parole, frasi spezzate, testo mancante, opzioni troncate).
4. Ha esattamente una risposta corretta chiaramente identificabile.
5. Le opzioni sono complete, leggibili e plausibili.
6. Non è banalmente ovvia (la risposta corretta non deve essere immediatamente
   evidente senza alcuna conoscenza specifica).

Rifiuta la domanda (REJECTED) in uno qualsiasi di questi casi:
- Riguarda scienze generali: biologia, fisica, chimica, astronomia, anatomia,
  informatica, matematica — ANCHE se scritta in italiano. Queste sono coperte
  da altri dataset.
- Riguarda fatti non italiani: geografia ungherese, storia croata, personaggi
  stranieri, trivia mondiale senza legame con l'Italia.
- È corrotta: contiene lettere casuali inserite nelle parole (es. "ZebraD",
  "BannoiAta", "contiente"), testo troncato, opzioni incomplete o senza senso.
- La domanda è incompleta o il testo della domanda non ha senso.
- Le opzioni non sono pertinenti o alcune sono chiaramente assurde (es. nomi
  di attori come opzioni a una domanda di anatomia).
- È sulla grammatica italiana solo se è talmente banale da essere inutile
  (es. "faccio" è indicativo presente — troppo elementare).
- Parla di eventi mondiali recenti senza specifico legame italiano.

Rispondi con esattamente una riga:
KEPT
oppure
REJECTED:<motivo_breve>

Non aggiungere altro testo."""


def _build_filter_message(prompt: str, answer: str, options: Optional[Dict] = None) -> str:
    if options:
        opts_str = "\n".join(f"{k}) {v}" for k, v in options.items())
        return (
            f"Domanda:\n{prompt}\n\n"
            f"Risposta corretta: {answer}\n\n"
            f"Valuta se questa MCQ deve essere KEPT o REJECTED."
        )
    return (
        f"Domanda (con opzioni incluse nel testo):\n{prompt}\n\n"
        f"Risposta corretta: {answer}\n\n"
        f"Valuta se questa MCQ deve essere KEPT o REJECTED."
    )

# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

async def _call_api(
    client: httpx.AsyncClient,
    model: str,
    messages: List[Dict],
    api_key: str,
    max_tokens: int,
    retries: int = DEFAULT_MAX_RETRIES,
) -> Optional[str]:
    payload = {
        "model":       model,
        "messages":    messages,
        "max_tokens":  max_tokens,
        "temperature": 0.0,
        "top_p":       1.0,
        "stream":      False,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for attempt in range(retries):
        try:
            resp = await client.post(
                f"{API_BASE}/chat/completions",
                json=payload, headers=headers, timeout=60.0,
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
# Parsing
# ---------------------------------------------------------------------------

def _parse_decision(raw: Optional[str]) -> tuple[str, str]:
    """Returns (decision, reason) where decision is 'KEPT', 'REJECTED', or 'UNSURE'."""
    if not raw:
        return "UNSURE", "api_failure"
    r = raw.strip()
    upper = r.upper()
    if upper == "KEPT":
        return "KEPT", ""
    if upper.startswith("REJECTED"):
        reason = r[8:].lstrip(":").strip() if len(r) > 8 else "no_reason"
        return "REJECTED", reason
    return "UNSURE", f"unexpected:{r[:60]}"

# ---------------------------------------------------------------------------
# Per-example filtering
# ---------------------------------------------------------------------------

async def _filter_example(
    ex: Dict[str, Any],
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    api_key: str,
) -> Dict[str, Any]:
    prompt = str(ex.get("prompt", "")).strip()
    answer = str(ex.get("answer", "")).strip()

    result = {
        **ex,
        "filter": {
            "pri": None,
            "sec": None,
            "final": None,
            "reason": None,
            "kept": False,
        }
    }

    messages = [
        {"role": "system", "content": SYSTEM_FILTER},
        {"role": "user",   "content": _build_filter_message(prompt, answer)},
    ]

    # Primary
    async with sem:
        pri_raw = await _call_api(client, TEACHER_PRI, messages, api_key, MAX_TOKENS_PRI)

    pri_dec, pri_reason = _parse_decision(pri_raw)
    result["filter"]["pri"] = {"raw": pri_raw, "decision": pri_dec, "reason": pri_reason}

    if pri_dec == "KEPT":
        result["filter"]["final"] = "KEPT"
        result["filter"]["kept"]  = True
        return result

    if pri_dec == "REJECTED":
        result["filter"]["final"]  = "REJECTED"
        result["filter"]["reason"] = pri_reason
        return result

    # UNSURE → secondary
    async with sem:
        sec_raw = await _call_api(client, TEACHER_SEC, messages, api_key, MAX_TOKENS_SEC)

    sec_dec, sec_reason = _parse_decision(sec_raw)
    result["filter"]["sec"] = {"raw": sec_raw, "decision": sec_dec, "reason": sec_reason}

    if sec_dec == "KEPT":
        result["filter"]["final"] = "KEPT"
        result["filter"]["kept"]  = True
    else:
        result["filter"]["final"]  = "REJECTED"
        result["filter"]["reason"] = sec_reason or pri_reason

    return result

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _load_done_ids(path: Path) -> set:
    done = set()
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    r = json.loads(line)
                    sid = str(r.get("source_id", r.get("id", ""))).strip()
                    if sid:
                        done.add(sid)
                except json.JSONDecodeError:
                    pass
    return done


def _append_jsonl(records: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


class _Progress:
    def __init__(self, total: int):
        self.total = total
        self.done = self.kept = self.rejected = 0
        self.t0 = time.time()

    def update(self, kept: bool) -> None:
        self.done += 1
        if kept:
            self.kept += 1
        else:
            self.rejected += 1
        if self.done % LOG_EVERY == 0 or self.done == self.total:
            elapsed = time.time() - self.t0
            rate    = self.done / elapsed if elapsed > 0 else 0
            eta     = (self.total - self.done) / rate if rate > 0 else 0
            pct     = 100 * self.kept // max(1, self.done)
            print(f"    {self.done}/{self.total} | kept={self.kept} ({pct}%) "
                  f"rejected={self.rejected} | {rate:.1f} ex/s | ETA {eta:.0f}s")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main_async(args):
    api_key = os.environ.get("CSCS_SERVING_API", "")
    if not api_key:
        raise ValueError("CSCS_SERVING_API not set.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix        = args.output_prefix or args.input.stem.replace("_parsed", "")
    kept_path     = args.output_dir / f"{prefix}.jsonl"
    rejected_path = args.output_dir / f"{prefix}_rejected.jsonl"
    all_path      = args.output_dir / f"{prefix}_filter_all.jsonl"

    records = _load_jsonl(args.input)
    print(f"Loaded {len(records)} examples from {args.input}")

    if args.dry_run:
        records = records[:args.debug_n]
        print(f"[DRY RUN] Processing {len(records)} examples")

    # Resume support
    done_ids = _load_done_ids(all_path)
    todo = [r for r in records
            if str(r.get("source_id", r.get("id", ""))).strip() not in done_ids]
    print(f"Already done: {len(done_ids)} | Remaining: {len(todo)}")

    existing_kept = 0
    if kept_path.exists():
        with open(kept_path, encoding="utf-8") as f:
            existing_kept = sum(1 for l in f if l.strip())
    print(f"Existing kept: {existing_kept}")

    sem  = asyncio.Semaphore(args.concurrency)
    prog = _Progress(len(todo))

    async def _run(ex):
        r = await _filter_example(ex, client, sem, api_key)
        prog.update(r["filter"]["kept"])
        return r

    async with httpx.AsyncClient() as client:
        for batch_start in range(0, len(todo), SAVE_EVERY):
            batch   = todo[batch_start:batch_start + SAVE_EVERY]
            results = await asyncio.gather(*[_run(ex) for ex in batch])

            kept_batch     = [r for r in results if r["filter"]["kept"]]
            rejected_batch = [r for r in results if not r["filter"]["kept"]]

            _append_jsonl(results,       all_path)
            _append_jsonl(kept_batch,    kept_path)
            _append_jsonl(rejected_batch, rejected_path)

    total_kept = existing_kept + prog.kept
    total_done = len(done_ids) + prog.done
    pct = 100 * total_kept // max(1, total_done)
    print(f"\nDone. Kept: {total_kept}/{total_done} ({pct}%)")
    print(f"  Kept:     {kept_path}")
    print(f"  Rejected: {rejected_path}")
    print(f"  All:      {all_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",       type=Path, required=True)
    p.add_argument("--output_dir",  type=Path,
                   default=Path("data/multilinguality/processed/polizia_di_stato/filtered"))
    p.add_argument("--concurrency", type=int,  default=DEFAULT_CONCURRENCY)
    p.add_argument("--dry_run",     action="store_true")
    p.add_argument("--debug_n",     type=int,  default=20)
    p.add_argument("--output_prefix", type=str, default=None,
               help="Prefix for output files. Defaults to input filename stem.")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Primary:    {TEACHER_PRI}")
    print(f"Secondary:  {TEACHER_SEC}")
    print(f"Input:      {args.input}")
    print(f"Output:     {args.output_dir}")
    print(f"Concurrency:{args.concurrency}")
    if args.dry_run:
        print(f"[DRY RUN] {args.debug_n} examples\n")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()