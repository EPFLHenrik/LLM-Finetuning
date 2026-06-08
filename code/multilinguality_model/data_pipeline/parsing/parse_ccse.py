"""code/multilinguality_model/data_pipeline/parsing/parse_ccse.py

Parse the Instituto Cervantes CCSE manual into processed Spanish MCQ JSONL.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pdfplumber

from multilinguality_model.paths import get_processed_data_dir
from multilinguality_model.data_pipeline.parsing.download_utils import ensure_raw_file


DATASET_NAME = "ccse"
LANGUAGE = "spanish"
COUNTRY = "Spain"
SOURCE = "Instituto Cervantes CCSE 2026"
SOURCE_URL = "https://examenes.cervantes.es/sites/default/files/manual-ccse-2026-def.pdf"

# The uploaded manual is 101 pages of visible content plus cover/metadata in PDF indexing.
# In pdfplumber 1-indexed page numbers, questions are on 18-98 and solutions on 99-101.
QUESTION_PAGE_MIN = 18
QUESTION_PAGE_MAX = 98
SOLUTION_PAGE_MIN = 99

QUESTION_START_RE = re.compile(r"^([1-5]\d{3})\s+(.+)$")
OPTION_RE = re.compile(r"^([abc])\.\s*(.*)$", re.IGNORECASE)
ANSWER_PAIR_RE = re.compile(r"\b([1-5]\d{3})\s+([abc])\b", re.IGNORECASE)

VALID_ANSWER_LETTERS = set("ABC")


TASK_METADATA = {
    "1": {
        "task": "TAREA 1",
        "subject": "government_legislation_citizenship",
        "subject_es": "Gobierno, legislación y participación ciudadana",
        "regionality": "regional",
        "difficulty": "easy_medium",
    },
    "2": {
        "task": "TAREA 2",
        "subject": "fundamental_rights_duties",
        "subject_es": "Derechos y deberes fundamentales",
        "regionality": "regional",
        "difficulty": "easy_medium",
    },
    "3": {
        "task": "TAREA 3",
        "subject": "spain_territorial_organization_geography",
        "subject_es": "Organización territorial de España. Geografía física y política",
        "regionality": "regional",
        "difficulty": "easy_medium",
    },
    "4": {
        "task": "TAREA 4",
        "subject": "spanish_culture_history",
        "subject_es": "Cultura e historia de España",
        "regionality": "regional",
        "difficulty": "easy_medium",
    },
    "5": {
        "task": "TAREA 5",
        "subject": "spanish_society",
        "subject_es": "Sociedad española",
        "regionality": "regional",
        "difficulty": "easy_medium",
    },
}


NOISE_EXACT = {
    "PREGUNTAS",
    "SATNUGERP",
    "SENOICULOS",
    "Manual de preparación de la prueba CCSE 2026",
    "Manual de preparación de la prueba CCSE",
    "Instituto Cervantes",
}

def is_valid_ccse_qid(qid: str) -> bool:
    n = int(qid)

    return (
        1001 <= n <= 1120
        or 2001 <= n <= 2036
        or 3001 <= n <= 3024
        or 4001 <= n <= 4036
        or 5001 <= n <= 5084
    )


def clean_line(line: str) -> str:
    line = line.replace("\ufeff", "")
    line = line.replace("\u00ad", "")
    line = line.replace("￾", "")
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def is_noise_line(line: str) -> bool:
    if not line:
        return True
    if line in NOISE_EXACT:
        return True
    if line.startswith("Manual de preparación"):
        return True
    if line.startswith("Tarea ") and "Solucionario" not in line:
        return True
    if re.fullmatch(r"\d+\s+Instituto Cervantes", line):
        return True
    if re.fullmatch(r"Instituto Cervantes\s+\d+", line):
        return True
    if re.fullmatch(r"\d{4}", line):
        # A bare four-digit number could be a question ID if followed by text.
        # Alone it is not a full question start.
        return True
    return False


def extract_question_lines(pdf_path: Path) -> List[Tuple[int, str]]:
    """Extract question pages column-wise to avoid two-column interleaving."""
    out: List[Tuple[int, str]] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            if page_no < QUESTION_PAGE_MIN or page_no > QUESTION_PAGE_MAX:
                continue

            split_x = page.width / 2.0
            boxes = [
                (0, 0, split_x, page.height),
                (split_x, 0, page.width, page.height),
            ]

            for box in boxes:
                text = page.crop(box).extract_text(x_tolerance=1.5, y_tolerance=3) or ""
                for raw in text.splitlines():
                    line = clean_line(raw)
                    if line:
                        out.append((page_no, line))

    return out


def parse_answer_key(pdf_path: Path) -> Dict[str, str]:
    answers: Dict[str, str] = {}

    with pdfplumber.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            if page_no < SOLUTION_PAGE_MIN:
                continue

            text = page.extract_text(x_tolerance=1.5, y_tolerance=3) or ""
            for qid, ans in ANSWER_PAIR_RE.findall(text):
                ans = ans.upper()
                if ans in VALID_ANSWER_LETTERS:
                    answers[qid] = ans

    return answers


def parse_questions(lines: List[Tuple[int, str]]) -> List[Dict[str, object]]:
    questions: List[Dict[str, object]] = []

    current: Optional[Dict[str, object]] = None
    current_option: Optional[str] = None

    def flush() -> None:
        nonlocal current, current_option

        if current is None:
            return

        option_lines = current.get("option_lines", {})
        question_lines = current.get("question_lines", [])

        if isinstance(option_lines, dict) and isinstance(question_lines, list) and len(option_lines) >= 2:
            question_text = " ".join(str(x).strip() for x in question_lines if str(x).strip()).strip()
            options = {
                letter: " ".join(parts).strip()
                for letter, parts in option_lines.items()
                if " ".join(parts).strip()
            }

            if question_text and len(options) >= 2:
                questions.append({
                    "id": current["id"],
                    "page": current["page"],
                    "question": question_text,
                    "options": options,
                })

        current = None
        current_option = None

    for page_no, raw_line in lines:
        line = clean_line(raw_line)

        if is_noise_line(line):
            continue

        m_q = QUESTION_START_RE.match(line)
        if m_q and is_valid_ccse_qid(m_q.group(1)):
            flush()
            qid = m_q.group(1)
            question_start = m_q.group(2).strip()
            current = {
                "id": qid,
                "page": page_no,
                "question_lines": [question_start],
                "option_lines": {},
            }
            current_option = None
            continue

        m_opt = OPTION_RE.match(line)
        if m_opt and current is not None:
            letter = m_opt.group(1).upper()
            text = m_opt.group(2).strip()
            current["option_lines"][letter] = [text]
            current_option = letter
            continue

        if current is not None:
            if current_option is not None:
                current["option_lines"][current_option].append(line)
            else:
                current["question_lines"].append(line)

    flush()
    return questions


def format_prompt(question: str, options: Dict[str, str]) -> str:
    options_text = "\n".join(f"{letter}) {text}" for letter, text in sorted(options.items()))
    return f"{question}\n\n{options_text}"


def to_processed_record(q: Dict[str, object], answer: str, idx: int) -> Dict[str, object]:
    qid = str(q["id"])
    task_id = qid[0]
    task_meta = TASK_METADATA.get(task_id, {})

    source_id = f"{DATASET_NAME}_{qid}"

    return {
        "id": f"{DATASET_NAME}_spanish_{idx}",
        "source_id": source_id,
        "dataset": DATASET_NAME,
        "language": LANGUAGE,
        "task_type": "mcq",
        "prompt": format_prompt(str(q["question"]), q["options"]),
        "answer": answer,
        "metadata": {
            "source": SOURCE,
            "source_url": SOURCE_URL,
            "source_country": COUNTRY,
            "source_exam": "CCSE",
            "source_question_id": qid,
            "source_page": q.get("page"),
            "n_options": len(q["options"]),
            "option_letters": sorted(q["options"].keys()),
            "task": task_meta.get("task"),
            "subject": task_meta.get("subject"),
            "subject_es": task_meta.get("subject_es"),
            "regionality": task_meta.get("regionality", "regional"),
            "difficulty": task_meta.get("difficulty", "easy_medium"),
            "license_note": "Instituto Cervantes CCSE preparation manual; verify reuse terms before redistribution.",
        },
    }


def validate_records(records: List[Dict[str, object]]) -> List[str]:
    warnings: List[str] = []

    expected_n = 299
    if len(records) != expected_n:
        warnings.append(f"Expected {expected_n} CCSE questions after dropping known bad items, found {len(records)}.")

    ids = [str(r["metadata"]["source_question_id"]) for r in records]
    duplicates = sorted({x for x in ids if ids.count(x) > 1})
    if duplicates:
        warnings.append(f"Duplicate source_question_id values: {duplicates}")

    for r in records:
        answer = str(r["answer"])
        prompt = str(r["prompt"])
        if f"{answer}) " not in prompt:
            warnings.append(f"Answer option {answer} not found in prompt for {r['source_id']}.")

    return warnings


def write_jsonl(records: Iterable[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default=None, help="Path to CCSE PDF.")
    parser.add_argument("--output", default=None, help="Output JSONL path.")
    parser.add_argument("--allow-count-mismatch", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    pdf_path = (
        Path(args.pdf)
        if args.pdf
        else ensure_raw_file("spanish_mcq/ccse-2026.pdf", download=not args.no_download)
    )
    output_path = (
        Path(args.output)
        if args.output
        else get_processed_data_dir() / "ccse" / "ccse.jsonl"
    )

    question_lines = extract_question_lines(pdf_path)
    questions = parse_questions(question_lines)
    answers = parse_answer_key(pdf_path)

    records: List[Dict[str, object]] = []
    missing_answers: List[str] = []

    DROP_SOURCE_IDS = {"ccse_4036"}

    for q in questions:
        qid = str(q["id"])
        answer = answers.get(qid)
        if answer is None:
            missing_answers.append(qid)
            continue

        record = to_processed_record(q, answer, len(records))

        if record["source_id"] in DROP_SOURCE_IDS:
            continue

        records.append(record)

    warnings = validate_records(records)

    if missing_answers:
        warnings.append(f"Missing answers for question IDs: {missing_answers}")

    if not args.allow_count_mismatch and warnings:
        print(json.dumps({
            "status": "validation_failed",
            "pdf": str(pdf_path),
            "n_questions": len(questions),
            "n_answers": len(answers),
            "n_records": len(records),
            "warnings": warnings,
        }, ensure_ascii=False, indent=2))
        raise SystemExit(1)

    write_jsonl(records, output_path)

    print(json.dumps({
        "status": "ok",
        "dataset": DATASET_NAME,
        "pdf": str(pdf_path),
        "output": str(output_path),
        "n_questions": len(questions),
        "n_answers": len(answers),
        "n_records": len(records),
        "warnings": warnings,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()