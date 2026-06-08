"""code/multilinguality_model/data_pipeline/generation/generate_rubq_mcq.py

Generate verified Russian regional/cultural MCQs from RuBQ samples.
"""

from __future__ import annotations
import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx
import random

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

API_BASE = "https://api.swissai.svc.cscs.ch/v1"
TEACHER_GEN = "meta-llama/Llama-3.3-70B-Instruct"
TEACHER_VER_1 = "meta-llama/Llama-3.3-70B-Instruct"
TEACHER_VER_2 = "swiss-ai/Apertus-70B-Instruct-2509"

MAX_TOKENS_GEN = 900
MAX_TOKENS_VER = 500

DEFAULT_CONCURRENCY = 8
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0
DEFAULT_GEN_TEMPERATURE = 0.65
VER_TEMPERATURE = 0.0
DEFAULT_TOP_P = 0.95
LOG_EVERY = 25
SAVE_EVERY = 25

DATASET = "rubq"
LANGUAGE = "russian"
LETTERS = ("A", "B", "C", "D")


SYSTEM_GEN = """Ты эксперт по российской, советской, русской и русскоязычной культуре.

Твоя задача — из одного примера RuBQ создать ОДИН качественный MCQ на русском языке.

Используй пример ТОЛЬКО если он действительно проверяет российское, советское, русское или русскоязычное культурное/региональное знание.

Подходят:
- русская и советская литература, кино, театр, музыка, искусство;
- российская/советская история, политика, общественные институты;
- география России, СССР, российских регионов, городов, республик и постсоветского пространства, если факт нетривиален;
- российские/советские деятели культуры, науки, спорта, политики;
- произведения, персонажи, здания, памятники, исторические названия и культурные объекты русскоязычного пространства.

Не подходят — верни SKIP:
- общемировая trivia без прямой связи с Россией/СССР/русскоязычной культурой;
- слишком простые вопросы вида «в какой стране находится известный город», если ответ очевиден;
- общая религия, биология, мифология, филология, бытовые факты;
- иностранные авторы, персонажи, города или события без прямой русской/советской связи;
- пример, где региональность возникает только потому, что вопрос написан по-русски;
- пример, где нужно искусственно добавить “русскую традицию” или “советский контекст”, хотя исходный факт сам по себе не региональный.

Требования к вопросу:
1. Вопрос должен быть трудным или хотя бы содержательно нетривиальным.
2. Не копируй исходный вопрос механически, если его можно улучшить.
3. Не раскрывай правильный ответ формулировкой вопроса.
4. Не добавляй неподтвержденные детали.
5. Если в исходных данных несколько правильных ответов, формулируй вопрос так, чтобы был ровно один правильный вариант; иначе верни SKIP.
6. Не превращай простой факт в “сложный” за счет лишних слов, если знание остается элементарным.

Требования к вариантам:
1. Должно быть ровно 4 варианта: A, B, C, D.
2. Только один вариант должен быть правильным.
3. Дистракторы должны быть правдоподобными и сложными.
4. Все варианты должны быть одного типа: писатели с писателями, города с городами, композиторы с композиторами, театры с театрами.
5. Не используй:
   - вымышленные или странно звучащие имена;
   - варианты другой категории;
   - очевидно абсурдные варианты;
   - слишком легкие варианты;
   - частично правильные варианты;
   - альтернативные имена, псевдонимы или полные/краткие формы правильного ответа.
6. Если правильный ответ — человек, не ставь среди дистракторов другое имя того же человека.
7. Если правильный ответ имеет известные варианты написания, не используй эти варианты как дистракторы.
8. Все варианты должны быть сопоставимы по длине, стилю и конкретности.

Классификация:
- Используй только "regional".
- Если пример не проходит строгий критерий региональности и качества, верни SKIP.

Формат вывода — строго JSON без дополнительного текста:
{
  "decision": "GENERATE",
  "regionality": "regional",
  "question": "...",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "answer": "A",
  "difficulty": "hard",
  "source_fact": "краткое описание факта"
}

Если пример не подходит, верни строго:
{
  "decision": "SKIP",
  "reason": "короткая причина"
}"""

SYSTEM_VER = """Ты строгий верификатор русскоязычных MCQ-вопросов для датасета о российском, советском, русском и русскоязычном культурном знании.

Тебе дан исходный пример RuBQ и сгенерированный MCQ.

Ответь VALID только если ВСЕ условия выполнены:
1. Вопрос основан на исходном примере RuBQ.
2. Правильный ответ напрямую подтверждается исходными данными.
3. Есть ровно один правильный ответ.
4. Вопрос явно требует российского, советского, русского или русскоязычного культурного/регионального знания.
5. Это не просто общемировая trivia на русском языке.
6. Вопрос не является слишком простым фактом вида «город находится в стране X», если это очевидное школьное знание.
7. Все дистракторы правдоподобны, но однозначно неправильны.
8. Дистракторы относятся к тому же типу сущности, что и правильный ответ.
9. Среди дистракторов нет альтернативного имени, псевдонима, полной формы, краткой формы или варианта написания правильного ответа.
10. Дистракторы не выглядят вымышленными или странными.
11. Варианты сопоставимы по длине, стилю и уровню конкретности.
12. Формулировка вопроса не делает правильный ответ очевидным.
13. Русский язык вопроса и вариантов грамматически корректен.

Сразу отклоняй:
- global trivia, ошибочно помеченную как regional;
- искусственно “регионализированные” общие факты;
- вопросы с несколькими правильными вариантами;
- вопросы, где один вариант является алиасом другого;
- вопросы с fake-sounding distractors;
- слишком тривиальные географические вопросы;
- вопросы с неподтвержденными добавленными деталями.

Ответь ровно одним из вариантов:
- VALID
- INVALID:not_russian_regional
- INVALID:too_trivial
- INVALID:answer_not_supported
- INVALID:multiple_correct_answers
- INVALID:alias_option
- INVALID:weak_distractors
- INVALID:fake_distractor
- INVALID:bad_russian
- INVALID:unsupported_added_detail
- NOT_SURE

Не добавляй никакого другого текста."""


def _stringify_answer(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, list):
        vals = []
        for item in x:
            if isinstance(item, dict):
                vals.append(str(item.get("label") or item.get("value") or item.get("answer") or item))
            else:
                vals.append(str(item))
        return "; ".join(v.strip() for v in vals if v and v.strip())
    if isinstance(x, dict):
        return str(x.get("label") or x.get("value") or x.get("answer") or x)
    return str(x)


def _first_present(record: Dict[str, Any], keys: List[str]) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return ""


def _source_id(record: Dict[str, Any], idx: int) -> str:
    for key in ("source_id", "uid", "id", "question_id", "question_uris"):
        value = record.get(key)
        if value not in (None, ""):
            return f"rubq_{str(value).strip()}"
    q = str(_first_present(record, ["question_text", "question", "query", "text"])).strip()
    raw = f"rubq\n{idx}\n{q}"
    return f"rubq_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def _extract_question(record: Dict[str, Any]) -> str:
    return str(_first_present(record, ["question_text", "question", "query", "text"])).strip()


def _extract_answer(record: Dict[str, Any]) -> str:
    value = _first_present(record, ["answers", "answer", "answer_text", "answer_texts", "answer_labels"])
    return _stringify_answer(value).strip()

def _has_duplicate_or_alias_options(options: dict) -> bool:
    vals = [str(v).strip().lower().replace("ё", "е") for v in options.values()]
    for i, a in enumerate(vals):
        for j, b in enumerate(vals):
            if i == j:
                continue
            if a and b and (a in b or b in a):
                return True
    return False


def _extract_context(record: Dict[str, Any]) -> str:
    fields = []
    for key in ("paragraph", "paragraphs", "context", "passage", "wiki_context", "sparql", "query_wikidata"):
        value = record.get(key)
        if value in (None, ""):
            continue
        if isinstance(value, list):
            value = "\n".join(_stringify_answer(v) for v in value)
        fields.append(f"{key}: {value}")
    return "\n\n".join(fields).strip()


def _build_gen_user_message(sample: Dict[str, Any]) -> str:
    question = sample["question"]
    answer = sample["answer"]
    context = sample.get("context", "")
    original = sample.get("raw", {})

    compact_meta = {}
    for key in ("question_uris", "answer_uris", "entities", "relations", "complexity", "template", "sparql"):
        if key in original:
            compact_meta[key] = original[key]

    return (
        f"Исходный вопрос RuBQ:\n{question}\n\n"
        f"Исходный правильный ответ/ответы:\n{answer}\n\n"
        f"Дополнительный контекст:\n{context if context else 'нет'}\n\n"
        f"Метаданные:\n{json.dumps(compact_meta, ensure_ascii=False)}\n\n"
        f"Создай один сложный и качественный MCQ по правилам."
    )


def _build_ver_user_message(sample: Dict[str, Any], mcq: Dict[str, Any]) -> str:
    opts = "\n".join(f"{k}) {v}" for k, v in mcq["options"].items())
    return (
        f"Исходный вопрос RuBQ:\n{sample['question']}\n\n"
        f"Исходный правильный ответ/ответы:\n{sample['answer']}\n\n"
        f"Дополнительный контекст:\n{sample.get('context', '') or 'нет'}\n\n"
        f"---\n"
        f"Сгенерированный вопрос:\n{mcq['question']}\n\n"
        f"{opts}\n\n"
        f"Указанный правильный ответ: {mcq['answer']}\n\n"
        f"Проверь качество MCQ."
    )


async def _call_api(
    client: httpx.AsyncClient,
    model: str,
    messages: List[Dict[str, str]],
    api_key: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    retries: int = DEFAULT_MAX_RETRIES,
) -> Optional[str]:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for attempt in range(retries):
        try:
            resp = await client.post(f"{API_BASE}/chat/completions", json=payload, headers=headers, timeout=180.0)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < retries - 1:
                await asyncio.sleep(DEFAULT_RETRY_DELAY * (attempt + 1))
            else:
                print(f"    [API error] {model}: {e}")
                return None
    return None


def _parse_json_object(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    if not raw:
        return None

    clean = raw.strip()
    clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s*```$", "", clean)

    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", clean, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None


def _parse_generation(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    parsed = _parse_json_object(raw)
    if not isinstance(parsed, dict):
        return None

    decision = str(parsed.get("decision", "")).strip().upper()
    if decision == "SKIP":
        return {"decision": "SKIP", "reason": str(parsed.get("reason", "")).strip()}

    if decision != "GENERATE":
        return None

    question = str(parsed.get("question", "")).strip()
    options = parsed.get("options", {})
    answer = str(parsed.get("answer", "")).strip().upper()
    regionality = str(parsed.get("regionality", "")).strip()

    if not question or not isinstance(options, dict):
        return None
    if not all(letter in options for letter in LETTERS):
        return None
    if answer not in LETTERS:
        return None
    if regionality !="regional":
        return {"decision": "SKIP", "reason": f"regionality_not_used:{regionality}"}

    clean_options = {letter: str(options[letter]).strip() for letter in LETTERS}
    if any(not value for value in clean_options.values()):
        return None
    if len(set(clean_options.values())) != 4:
        return None
    
    if _has_duplicate_or_alias_options(clean_options):
        return None

    return {
        "decision": "GENERATE",
        "regionality": regionality,
        "question": question,
        "options": clean_options,
        "answer": answer,
        "difficulty": str(parsed.get("difficulty", "hard")).strip() or "hard",
        "source_fact": str(parsed.get("source_fact", "")).strip(),
    }


def _parse_verification(raw: Optional[str]) -> str:
    if not raw:
        return "api_failure"

    r = raw.strip()
    upper = r.upper()

    if upper == "VALID":
        return "VALID"
    if upper == "NOT_SURE":
        return "NOT_SURE"
    if upper.startswith("INVALID"):
        return r

    return f"unexpected_output:{r[:80]}"


def _shuffle_options(mcq: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    values = [mcq["options"][letter] for letter in LETTERS]
    correct = mcq["options"][mcq["answer"]]
    rng.shuffle(values)
    new_options = dict(zip(LETTERS, values))
    new_answer = next(letter for letter, value in new_options.items() if value == correct)
    return {**mcq, "options": new_options, "answer": new_answer}


def _build_prompt(mcq: Dict[str, Any]) -> str:
    options = "\n".join(f"{letter}) {mcq['options'][letter]}" for letter in LETTERS)
    return f"{mcq['question']}\n\n{options}"


async def _process_sample(
    sample: Dict[str, Any],
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    api_key: str,
    gen_temperature: float,
    top_p: float,
) -> Dict[str, Any]:
    source_id = sample["source_id"]

    result = {
        "source_id": source_id,
        "dataset": DATASET,
        "language": LANGUAGE,
        "task_type": "mcq",
        "rubq": {
            "question": sample["question"],
            "answer": sample["answer"],
            "context": sample.get("context", ""),
            "raw": sample.get("raw", {}),
        },
        "mcq": None,
        "generation": {"raw": None, "parsed": None, "valid": False, "reason": None},
        "verification": {"ver1": None, "ver2": None, "final": None},
        "valid": False,
        "prompt": None,
        "answer": None,
        "metadata": {
            "source_dataset": "RuBQ",
            "source_question": sample["question"],
            "source_answer": sample["answer"],
            "license": "CC-BY-SA-4.0",
            "generation_model": TEACHER_GEN,
            "verification_model1": TEACHER_VER_1,
            "verification_model2": TEACHER_VER_2,
        },
    }

    if not sample["question"] or not sample["answer"]:
        result["generation"]["reason"] = "missing_question_or_answer"
        return result

    gen_messages = [
        {"role": "system", "content": SYSTEM_GEN},
        {"role": "user", "content": _build_gen_user_message(sample)},
    ]

    async with sem:
        gen_raw = await _call_api(client, TEACHER_GEN, gen_messages, api_key, MAX_TOKENS_GEN, gen_temperature, top_p)

    result["generation"]["raw"] = gen_raw
    mcq = _parse_generation(gen_raw)

    if mcq is None:
        result["generation"]["reason"] = "parse_failed"
        return result

    if mcq.get("decision") == "SKIP":
        result["generation"]["reason"] = mcq.get("reason", "skipped_by_generator")
        return result

    rng = random.Random(source_id)
    mcq = _shuffle_options(mcq, rng)

    result["generation"]["parsed"] = mcq
    result["generation"]["valid"] = True
    result["mcq"] = mcq
    result["metadata"]["regionality"] = mcq["regionality"]
    result["metadata"]["difficulty"] = mcq.get("difficulty", "hard")
    result["metadata"]["source_fact"] = mcq.get("source_fact", "")

    ver_messages = [
        {"role": "system", "content": SYSTEM_VER},
        {"role": "user", "content": _build_ver_user_message(sample, mcq)},
    ]

    async with sem:
        ver1_raw = await _call_api(client, TEACHER_VER_1, ver_messages, api_key, MAX_TOKENS_VER, VER_TEMPERATURE, top_p)

    ver1_result = _parse_verification(ver1_raw)
    result["verification"]["ver1"] = {"raw": ver1_raw, "result": ver1_result}

    if ver1_result == "VALID":
        result["verification"]["final"] = "VALID"
        result["valid"] = True

    elif ver1_result in {"NOT_SURE", "api_failure"}:
        async with sem:
            ver2_raw = await _call_api(client, TEACHER_VER_2, ver_messages, api_key, MAX_TOKENS_VER, VER_TEMPERATURE, top_p)

        ver2_result = _parse_verification(ver2_raw)
        result["verification"]["ver2"] = {"raw": ver2_raw, "result": ver2_result}
        result["verification"]["final"] = ver2_result

        if ver2_result == "VALID":
            result["valid"] = True
        else:
            result["generation"]["reason"] = f"verification_failed:{ver2_result}"

    else:
        result["verification"]["final"] = ver1_result
        result["generation"]["reason"] = f"verification_failed:{ver1_result}"

    if result["valid"]:
        result["prompt"] = _build_prompt(mcq)
        result["answer"] = mcq["answer"]

    return result


def _load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("questions", "data", "items", "records"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError(f"Could not infer records list from {path}")


def _normalize_samples(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    samples = []

    for idx, record in enumerate(records):
        question = _extract_question(record)
        answer = _extract_answer(record)
        context = _extract_context(record)
        source_id = _source_id(record, idx)

        if not question or not answer:
            continue

        samples.append({
            "source_id": source_id,
            "question": question,
            "answer": answer,
            "context": context,
            "raw": record,
        })

    return samples


def _append_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_done_ids(path: Path) -> set:
    done = set()
    if not path.exists():
        return done

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                sid = str(record.get("source_id", "")).strip()
                if sid:
                    done.add(sid)
            except json.JSONDecodeError:
                pass

    return done


class _Progress:
    def __init__(self, total: int, log_every: int):
        self.total = total
        self.log_every = log_every
        self.done = 0
        self.valid = 0
        self.t0 = time.time()

    def update(self, valid: bool) -> None:
        self.done += 1
        if valid:
            self.valid += 1

        if self.done % self.log_every == 0 or self.done == self.total:
            elapsed = time.time() - self.t0
            rate = self.done / elapsed if elapsed > 0 else 0.0
            eta = (self.total - self.done) / rate if rate > 0 else 0.0
            pct = 100 * self.valid // max(1, self.done)
            print(f"    {self.done}/{self.total} | valid={self.valid} ({pct}%) | {rate:.1f} ex/s | ETA {eta:.0f}s")


async def main_async(args):
    api_key = os.environ.get("CSCS_SERVING_API", "")
    if not api_key:
        raise ValueError("CSCS_SERVING_API not set.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_path = args.output_dir / "rubq_all.jsonl"
    valid_path = args.output_dir / "rubq_generated.jsonl"
    fail_path = args.output_dir / "rubq_fail.jsonl"

    raw_records = _load_json_or_jsonl(args.input)
    samples = _normalize_samples(raw_records)

    if args.max_samples is not None:
        samples = samples[:args.max_samples]

    if args.dry_run:
        samples = samples[:args.debug_n]

    done_ids = _load_done_ids(all_path)
    todo = [sample for sample in samples if sample["source_id"] not in done_ids]

    print(f"Loaded raw records: {len(raw_records)}")
    print(f"Normalized usable samples: {len(samples)}")
    print(f"Already processed: {len(done_ids)}")
    print(f"Remaining: {len(todo)}")

    existing_valid = 0
    if all_path.exists():
        with all_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        existing_valid += int(bool(json.loads(line).get("valid")))
                    except json.JSONDecodeError:
                        pass

    if args.max_valid and existing_valid >= args.max_valid:
        print(f"Valid cap already reached: {existing_valid}/{args.max_valid}")
        return

    sem = asyncio.Semaphore(args.concurrency)
    progress = _Progress(len(todo), LOG_EVERY)

    async with httpx.AsyncClient() as client:
        for batch_start in range(0, len(todo), SAVE_EVERY):
            if args.max_valid and existing_valid >= args.max_valid:
                print(f"Valid cap reached: {existing_valid}/{args.max_valid}")
                break

            batch = todo[batch_start:batch_start + SAVE_EVERY]

            async def _run(sample):
                result = await _process_sample(sample, client, sem, api_key, args.gen_temperature, args.top_p)
                progress.update(result["valid"])
                return result

            results = await asyncio.gather(*[_run(sample) for sample in batch])

            valid_batch = [r for r in results if r["valid"]]
            fail_batch = [r for r in results if not r["valid"]]

            if args.max_valid:
                kept_valid = []
                for record in valid_batch:
                    if existing_valid >= args.max_valid:
                        break
                    kept_valid.append(record)
                    existing_valid += 1
                valid_batch = kept_valid
            else:
                existing_valid += len(valid_batch)

            _append_jsonl(results, all_path)
            _append_jsonl(valid_batch, valid_path)
            _append_jsonl(fail_batch, fail_path)

    total_processed = len(done_ids) + len(todo)
    pct = 100 * existing_valid // max(1, total_processed)

    print()
    print(f"Done. Valid: {existing_valid}/{total_processed} ({pct}%)")
    print(f"  Generated: {valid_path}")
    print(f"  All:       {all_path}")
    print(f"  Fail:      {fail_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, default=Path("data/multilinguality/processed/rubq"))
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    p.add_argument("--gen_temperature", type=float, default=DEFAULT_GEN_TEMPERATURE)
    p.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    p.add_argument("--max_valid", type=int, default=None)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--debug_n", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Generation model:      {TEACHER_GEN}")
    print(f"Verification model 1:  {TEACHER_VER_1}")
    print(f"Verification model 2:  {TEACHER_VER_2}")
    print(f"Input:                 {args.input}")
    print(f"Output:                {args.output_dir}")
    print(f"Concurrency:           {args.concurrency}")
    print(f"Gen temperature:       {args.gen_temperature}")
    print(f"Top p:                 {args.top_p}")

    if args.dry_run:
        print(f"[DRY RUN] {args.debug_n} samples")

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()