"""multilingual/data_pipeline/generation/generate_spain_boe_mcq.py

Generates Spanish MCQ questions from BOE legal articles (Constitution +
Autonomous Community Statutes) using a two-stage pipeline:

Stage 1 — Generation (Llama):
  Read a legal article with its full hierarchical context and generate one
  4-option MCQ testing substantive legal/regional knowledge.

Stage 2 — Verification (Llama primary, Apertus secondary):
  Verify that the correct answer is directly supported by the article text,
  distractors are clearly wrong, and no ambiguity exists.
  Output: VALID, INVALID:<reason>, or NOT_SURE.

Saving is append-only and resumable:
    <output_dir>/<dataset>_all.jsonl       ← all attempts
    <output_dir>/<dataset>_generated.jsonl ← verified valid MCQs
    <output_dir>/<dataset>_fail.jsonl      ← rejected examples

Supports processing a single dataset file or all files in a directory.

Usage:
    export CSCS_SERVING_API="sk-rc-..."

    # Single file
    python -m multilingual.data_pipeline.generation.generate_spain_boe_mcq \\
        --input multilingual/data/raw/spain_boe_legislation/spain_boe_constitucion_espanola.jsonl \\
        --output_dir multilingual/data/processed/spain_boe_constitucion_espanola \\
        --concurrency 20

    # All files in directory
    python -m multilingual.data_pipeline.generation.generate_spain_boe_mcq \\
        --input_dir multilingual/data/raw/spain_boe_legislation \\
        --output_base multilingual/data/processed \\
        --concurrency 20 \\
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

MAX_TOKENS_GEN    = 700
MAX_TOKENS_VER    = 400

DEFAULT_CONCURRENCY     = 8
DEFAULT_MAX_RETRIES     = 3
DEFAULT_RETRY_DELAY     = 1.0
DEFAULT_GEN_TEMPERATURE = 0.45
VER_TEMPERATURE         = 0.0
DEFAULT_TOP_P           = 0.95
LOG_EVERY               = 25
SAVE_EVERY              = 25
MIN_ARTICLE_CHARS       = 80

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_GEN = """Eres un experto en derecho constitucional español, Estatutos de Autonomía y elaboración de preguntas jurídicas tipo oposición.

Tu tarea es generar UNA sola pregunta MCQ en español a partir de un artículo legal oficial.

Objetivo:
Crear una pregunta jurídicamente precisa, de dificultad media o media-alta, con una sola respuesta correcta y tres distractores plausibles pero claramente erróneos.

Usa el artículo solo si contiene una regla, competencia, institución, procedimiento, derecho, principio, condición, excepción, plazo, mayoría, órgano o efecto jurídico suficientemente claro. Si el artículo es demasiado breve, puramente declarativo, demasiado genérico, corrupto, incompleto o no permite una buena pregunta, responde SKIP.

Criterios para una buena pregunta:
1. Debe ser respondible directamente a partir del artículo.
2. Debe evaluar un elemento jurídico sustantivo: órgano competente, alcance de una competencia, condición, límite, procedimiento, mayoría, efecto, derecho, deber o excepción.
3. No preguntes simplemente “qué dice el artículo” ni hagas una pregunta de copia literal.
4. Evita preguntas demasiado fáciles si el artículo permite una formulación más precisa.
5. No uses conocimiento externo al artículo.
6. Para Estatutos de Autonomía, prioriza preguntas que dependan de la comunidad autónoma concreta: instituciones propias, lengua, territorio, competencias, relaciones con el Estado o la UE, hacienda, organización institucional o régimen especial.
7. Para la Constitución, prioriza reglas institucionales o procedimentales precisas, no definiciones triviales.

Criterios para la respuesta correcta:
1. Debe estar directamente respaldada por el texto.
2. Debe ser jurídicamente exacta.
3. No debe ser mucho más larga, completa o técnica que los distractores.
4. No debe ser la única opción con redacción legal natural.
5. Si el artículo contiene varios elementos, formula la pregunta para que solo uno sea necesario y claramente correcto.

Criterios para los distractores:
1. Deben ser plausibles para alguien que estudia derecho constitucional o autonómico.
2. Deben pertenecer a la misma categoría jurídica que la respuesta correcta:
   - órganos con órganos;
   - competencias con competencias;
   - mayorías con mayorías;
   - procedimientos con procedimientos;
   - derechos con derechos;
   - efectos jurídicos con efectos jurídicos.
3. Deben modificar un elemento jurídicamente relevante: órgano, plazo, mayoría, ámbito territorial, condición, límite, tipo de norma, competencia o efecto.
4. No uses distractores absurdos, caricaturescos o evidentemente imposibles.
5. No uses opciones como “crear organismos internacionales”, “dictar normas de derecho internacional” o fórmulas extremas similares salvo que sean realmente plausibles en el contexto.
6. No uses distractores que sean solo información ausente: deben ser incompatibles con el artículo.
7. No generes opciones parcialmente correctas.
8. Todas las opciones deben ser comparables en longitud, estilo y nivel de detalle.
9. Si la respuesta correcta es larga, los distractores también deben ser largos y jurídicamente estructurados.

Evita especialmente:
- respuestas correctas que sean la única opción larga;
- distractores evidentemente ridículos;
- preguntas triviales sobre capitales, nombres o definiciones si el artículo permite algo más sustantivo;
- preguntas demasiado amplias que obliguen a copiar medio artículo;
- opciones con “siempre”, “nunca”, “exclusivamente” o “sin limitación” cuando sean demasiado fáciles.

Formato de salida: JSON puro, sin texto adicional:
{
  "question": "...",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "answer": "A"
}

Si el artículo no permite una buena pregunta, responde exactamente:
SKIP"""

SYSTEM_VER = """Eres un verificador severo de preguntas MCQ de derecho constitucional español y derecho autonómico.

Se te proporcionará:
1. El texto original del artículo legal.
2. Una pregunta MCQ con cuatro opciones A, B, C, D.
3. La respuesta correcta indicada.

Responde VALID solo si TODOS los criterios se cumplen:
1. La pregunta está directamente fundada en el artículo.
2. La respuesta correcta está clara y directamente respaldada por el texto.
3. Existe una sola respuesta correcta.
4. Ningún distractor es parcialmente correcto.
5. Los distractores son plausibles para un estudiante de derecho, pero incompatibles con el artículo.
6. Los distractores modifican elementos jurídicamente relevantes, no detalles aleatorios.
7. Todas las opciones pertenecen a la misma categoría jurídica.
8. Las opciones son comparables en longitud, estilo y especificidad.
9. La respuesta correcta no es obvia por ser la única opción larga, técnica o completa.
10. La pregunta no es excesivamente trivial si el artículo permitía una pregunta más sustantiva.
11. La pregunta no requiere conocimiento externo al artículo.
12. La redacción es natural, formal y gramaticalmente correcta en español.

Rechaza inmediatamente si:
- la respuesta correcta no está respaldada;
- hay varias respuestas correctas o parcialmente correctas;
- los distractores son absurdos, caricaturescos o claramente imposibles;
- los distractores solo están ausentes del texto pero no contradicen el artículo;
- la respuesta correcta es la única opción jurídicamente precisa;
- la pregunta se limita a copiar el artículo de forma mecánica;
- la pregunta es demasiado fácil, por ejemplo una capital, un nombre obvio o una institución sin ninguna condición jurídica relevante;
- el artículo es demasiado breve, genérico, corrupto o insuficiente;
- el texto exige conocimiento externo.

Output permitido, exactamente uno:
- VALID
- INVALID:respuesta_no_respaldada
- INVALID:varias_respuestas_correctas
- INVALID:distractores_parcialmente_correctos
- INVALID:distractores_debiles
- INVALID:distractores_absurdos
- INVALID:opciones_desbalanceadas
- INVALID:demasiado_trivial
- INVALID:pregunta_demasiado_mecanica
- INVALID:articulo_inadecuado
- INVALID:requiere_conocimiento_externo
- INVALID:espanol_no_natural
- NOT_SURE

No añadas ningún otro texto."""


def _build_context(article: Dict[str, Any]) -> str:
    """Build a readable hierarchy string from título/capítulo/sección/subsección."""
    parts = []
    for field in ("título", "capítulo", "sección", "subsección"):
        val = article.get(field) or (article.get("metadata") or {}).get(field)
        if val:
            parts.append(val)
    return " > ".join(parts) if parts else ""


def _gen_user_message(article: Dict[str, Any]) -> str:
    law_name   = article.get("law_name", "")
    region     = article.get("region") or (article.get("metadata") or {}).get("region")
    art_num    = article.get("article_number", "")
    art_title  = article.get("article_title") or ""
    source_text = article.get("source_text", "")
    context    = _build_context(article)

    region_line = f"Comunidad Autónoma: {region}\n" if region else ""
    title_line  = f"Título del artículo: {art_title}\n" if art_title else ""
    ctx_line    = f"Contexto jerárquico: {context}\n" if context else ""

    return (
        f"Ley: {law_name}\n"
        f"{region_line}"
        f"{ctx_line}"
        f"Artículo {art_num}{': ' + art_title if art_title else ''}\n\n"
        f"Texto del artículo:\n{source_text}\n\n"
        f"Genera una pregunta MCQ a partir de este artículo."
    )


def _ver_user_message(
    article: Dict[str, Any],
    question: str,
    options: Dict[str, str],
    answer: str,
) -> str:
    law_name    = article.get("law_name", "")
    art_num     = article.get("article_number", "")
    art_title   = article.get("article_title") or ""
    source_text = article.get("source_text", "")
    opts        = "\n".join(f"{k}) {v}" for k, v in options.items())

    return (
        f"Ley: {law_name} — Artículo {art_num}{': ' + art_title if art_title else ''}\n\n"
        f"Texto original:\n{source_text}\n\n"
        f"---\n"
        f"Pregunta MCQ generada:\n{question}\n\n"
        f"{opts}\n\n"
        f"Respuesta correcta indicada: {answer}\n\n"
        f"Verifica la calidad de esta MCQ según los criterios proporcionados."
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
        return raw.strip()
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
    source_id = article.get("source_id") or article.get("id", "")
    dataset   = article.get("dataset", "spain_boe")
    meta      = article.get("metadata", {})

    result = {
        "source_id":  source_id,
        "dataset":    dataset,
        "language":   "spanish",
        "task_type":  "mcq",
        "mcq":        None,
        "generation":  {"raw": None, "parsed": None, "valid": False, "reason": None},
        "verification": {"ver1": None, "ver2": None, "final": None},
        "valid":      False,
        "prompt":     None,
        "answer":     None,
        "metadata": {
            "country":          article.get("country", "Spain"),
            "region":           article.get("region") or meta.get("region"),
            "regionality":      meta.get("regionality", "regional_explicit" if article.get("region") else "region_agnostic"),
            "law_name":         article.get("law_name") or meta.get("law_name"),
            "source_url":       article.get("source_url") or meta.get("source_url"),
            "license":          article.get("license") or meta.get("license"),
            "source_unit_type": "article",
            "boe_id":           meta.get("boe_id"),
            "título":           article.get("título") or meta.get("título"),
            "capítulo":         article.get("capítulo") or meta.get("capítulo"),
            "sección":          article.get("sección") or meta.get("sección"),
            "subsección":       article.get("subsección") or meta.get("subsección"),
            "article_number":   article.get("article_number"),
            "article_title":    article.get("article_title"),
            "generation_model":    TEACHER_GEN,
            "verification_model1": TEACHER_VER_1,
            "verification_model2": TEACHER_VER_2,
        },
    }

    source_text = article.get("source_text", "").strip()
    if len(source_text) < MIN_ARTICLE_CHARS:
        result["generation"]["reason"] = "too_short"
        return result

    # ── Stage 1: Generation ──────────────────────────────────────────────────
    gen_messages = [
        {"role": "system", "content": SYSTEM_GEN},
        {"role": "user",   "content": _gen_user_message(article)},
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

    rng = random.Random(source_id)
    mcq = _shuffle_options(mcq, rng)

    result["generation"]["parsed"] = mcq
    result["generation"]["valid"]  = True
    result["mcq"] = mcq

    # ── Stage 2a: Verification — Llama ──────────────────────────────────────
    ver_messages = [
        {"role": "system", "content": SYSTEM_VER},
        {"role": "user",   "content": _ver_user_message(
            article, mcq["question"], mcq["options"], mcq["answer"],
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


def _load_articles(input_path: Path) -> List[Dict[str, Any]]:
    """Load source articles from a JSONL file, filtering too-short ones."""
    articles = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                if len(r.get("source_text", "").strip()) >= MIN_ARTICLE_CHARS:
                    articles.append(r)
            except json.JSONDecodeError:
                pass
    return articles

# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

async def _process_file(
    input_path: Path,
    output_dir: Path,
    args,
    api_key: str,
    combined_path: Path | None = None,
) -> None:
    dataset = input_path.stem  # e.g. "spain_boe_constitucion_espanola"
    all_path   = output_dir / f"{dataset}_all.jsonl"
    valid_path = output_dir / f"{dataset}_generated.jsonl"
    fail_path  = output_dir / f"{dataset}_fail.jsonl"

    print(f"\n{'='*60}")
    print(f"Processing: {input_path.name}")
    print(f"Output:     {output_dir}")

    articles = _load_articles(input_path)
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
                        if json.loads(line).get("valid"):
                            existing_valid += 1
                    except json.JSONDecodeError:
                        pass

    todo = [a for a in articles if (a.get("source_id") or a.get("id", "")) not in done_ids]
    print(f"Articles: {len(todo)} remaining / {len(articles)} total")

    if not todo:
        print("Nothing to do.")
        return

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
            batch   = todo[batch_start:batch_start + SAVE_EVERY]
            results = await asyncio.gather(*[_run(a) for a in batch])

            valid_batch = [r for r in results if r["valid"]]
            fail_batch  = [r for r in results if not r["valid"]]
            valid_so_far += len(valid_batch)

            _append_jsonl(results,     all_path)
            _append_jsonl(valid_batch, valid_path)
            _append_jsonl(fail_batch,  fail_path)
            if combined_path is not None:
                _append_jsonl(valid_batch, combined_path)

    total = len(done_ids) + len(todo)
    pct   = 100 * valid_so_far // max(1, total)
    print(f"Done. Valid: {valid_so_far}/{total} ({pct}%)")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main_async(args):
    api_key = os.environ.get("CSCS_SERVING_API", "")
    if not api_key:
        raise ValueError("CSCS_SERVING_API not set.")

    if args.input:
        # Single file mode
        input_path = Path(args.input)
        output_dir = Path(args.output_dir)
        await _process_file(input_path, output_dir, args, api_key)
    else:
        # Directory mode — process all .jsonl files
        input_dir  = Path(args.input_dir)
        output_base = Path(args.output_base)
        files = sorted(input_dir.glob("*.jsonl"))
        print(f"Found {len(files)} JSONL files in {input_dir}")
        output_dir = output_base / "spain_boe_legislation_strict"
        combined_path = output_dir / "spain_boe_legislation_generated.jsonl"
        for f in files:
            await _process_file(f, output_dir, args, api_key, combined_path=combined_path)


def parse_args():
    p = argparse.ArgumentParser()

    # Input — single file or directory
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--input",     type=str, help="Single input JSONL file")
    grp.add_argument("--input_dir", type=str, help="Directory of JSONL files to process all")

    # Output
    p.add_argument("--output_dir",  type=str, default=None,
                   help="Output dir (single file mode)")
    p.add_argument("--output_base", type=str,
                   default="multilingual/data/processed",
                   help="Base output dir (directory mode — one subdir per dataset)")

    p.add_argument("--concurrency",     type=int,   default=DEFAULT_CONCURRENCY)
    p.add_argument("--gen_temperature", type=float, default=DEFAULT_GEN_TEMPERATURE)
    p.add_argument("--top_p",           type=float, default=DEFAULT_TOP_P)
    p.add_argument("--dry_run",         action="store_true")
    p.add_argument("--debug_n",         type=int,   default=5)
    return p.parse_args()


def main():
    args = parse_args()

    if args.input and not args.output_dir:
        args.output_dir = str(
            Path("multilingual/data/processed") / Path(args.input).stem
        )

    print(f"Generation model:     {TEACHER_GEN}")
    print(f"Verification model 1: {TEACHER_VER_1}")
    print(f"Verification model 2: {TEACHER_VER_2}")
    print(f"Concurrency:          {args.concurrency}")
    print(f"Gen temperature:      {args.gen_temperature}")
    if args.dry_run:
        print(f"[DRY RUN] {args.debug_n} articles per file\n")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()