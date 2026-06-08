"""code/multilinguality_model/data_pipeline/parsing/parse_demre.py

Parse DEMRE PAES Historia questions into processed Spanish MCQ JSONL.
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


DATASET_NAME = "demre"
LANGUAGE = "spanish"
COUNTRY = "Chile"
SOURCE = "DEMRE PAES Historia y Ciencias Sociales Admisión 2025"
SOURCE_URL = "https://demre.cl/"
SOURCE_FORM = "Forma 123"

QUESTION_PAGE_MIN = 3
QUESTION_PAGE_MAX = 36

QUESTION_NUM_RE = re.compile(r"^(\d{1,3})\.$")
OPTION_RE = re.compile(r"^([A-E])\)\s*(.+)$")
ANSWER_PAIR_RE = re.compile(r"\b(\d{1,3})\*?\s+([A-E])\b", re.IGNORECASE)

SECTION_HEADERS = {
    "FORMACIÓN CIUDADANA": {
        "subject": "citizenship",
        "subject_es": "Formación ciudadana",
        "regionality": "regional",
    },
    "HISTORIA: MUNDO, AMÉRICA Y CHILE": {
        "subject": "history_world_america_chile",
        "subject_es": "Historia: mundo, América y Chile",
        "regionality": "regional",
    },
    "GEOGRAFÍA": {
        "subject": "geography",
        "subject_es": "Geografía",
        "regionality": "regional",
    },
    "ECONOMÍA Y SOCIEDAD": {
        "subject": "economics_society",
        "subject_es": "Economía y sociedad",
        "regionality": "regional",
    },
    "SISTEMA ECONÓMICO": {
        "subject": "economics_society",
        "subject_es": "Sistema económico",
        "regionality": "regional",
    },
}


def clean_line(line: str) -> str:
    line = line.replace("\ufeff", "")
    line = line.replace("\u00ad", "")
    line = line.replace("￾", "")
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def is_noise_line(line: str) -> bool:
    if not line:
        return True
    if line == "FORMA 123 - 2025":
        return True
    if re.fullmatch(r"-\s*\d+\s*-", line):
        return True
    if line.startswith("Registro de Propiedad Intelectual"):
        return True
    if line.startswith("Universidad de Chile"):
        return True
    if "Derechos reservados" in line:
        return True
    if line.startswith("Listado de Referencias Bibliográficas"):
        return True
    if line.startswith("N° de ítem"):
        return True
    return False


def extract_lines(pdf_path: Path) -> List[Tuple[int, str]]:
    lines: List[Tuple[int, str]] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            if page_no < QUESTION_PAGE_MIN or page_no > QUESTION_PAGE_MAX:
                continue

            text = page.extract_text(x_tolerance=1.5, y_tolerance=3) or ""
            for raw in text.splitlines():
                line = clean_line(raw)
                if line:
                    lines.append((page_no, line))

    return lines


def parse_questions(lines: List[Tuple[int, str]]) -> List[Dict[str, object]]:
    questions: List[Dict[str, object]] = []

    current_section: Optional[str] = None
    current: Optional[Dict[str, object]] = None
    current_option: Optional[str] = None

    def flush() -> None:
        nonlocal current, current_option

        if current is None:
            return

        question_lines = current.get("question_lines", [])
        option_lines = current.get("option_lines", {})

        if isinstance(question_lines, list) and isinstance(option_lines, dict) and len(option_lines) >= 2:
            question = " ".join(str(x).strip() for x in question_lines if str(x).strip()).strip()
            options = {
                letter: " ".join(parts).strip()
                for letter, parts in option_lines.items()
                if " ".join(parts).strip()
            }

            if question and len(options) >= 2:
                questions.append({
                    "id": current["id"],
                    "page": current["page"],
                    "section": current.get("section"),
                    "question": question,
                    "options": options,
                })

        current = None
        current_option = None

    for page_no, raw_line in lines:
        line = clean_line(raw_line)

        if is_noise_line(line):
            continue

        if line in SECTION_HEADERS:
            flush()
            current_section = line
            continue

        m_q = QUESTION_NUM_RE.match(line)
        if m_q:
            flush()
            current = {
                "id": m_q.group(1),
                "page": page_no,
                "section": current_section,
                "question_lines": [],
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


def parse_answers_from_pdf(answer_pdf: Path) -> Dict[str, str]:
    """Parse the DEMRE clavijero answer PDF.

    The answer-key PDF has a cover page and later score-transformation tables.
    We only parse pages containing the CLAVES table, and allow optional '*' markers
    after item numbers.
    """
    answers: Dict[str, str] = {}

    with pdfplumber.open(answer_pdf) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=1.5, y_tolerance=3) or ""

            if "CLAVES" not in text.upper():
                continue

            for qid, ans in ANSWER_PAIR_RE.findall(text):
                answers[str(int(qid))] = ans.upper()

    return answers


def parse_answers_from_json(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError("--answers-json must be a JSON object mapping question IDs to answer letters.")

    out: Dict[str, str] = {}
    for k, v in raw.items():
        qid = str(int(str(k).strip()))
        ans = str(v).strip().upper()
        if ans not in {"A", "B", "C", "D", "E"}:
            raise ValueError(f"Invalid answer for question {qid}: {v!r}")
        out[qid] = ans

    return out


def load_answers(answer_pdf: Optional[Path], answers_json: Optional[Path]) -> Dict[str, str]:
    if answer_pdf is not None and answers_json is not None:
        raise ValueError("Use only one of --answer-pdf or --answers-json, not both.")

    if answer_pdf is not None:
        return parse_answers_from_pdf(answer_pdf)

    if answers_json is not None:
        return parse_answers_from_json(answers_json)

    raise ValueError("DEMRE parsing requires --answer-pdf or --answers-json.")


def format_prompt(question: str, options: Dict[str, str]) -> str:
    options_text = "\n".join(f"{letter}) {text}" for letter, text in sorted(options.items()))
    return f"{question}\n\n{options_text}"


def to_processed_record(q: Dict[str, object], answer: str, idx: int) -> Dict[str, object]:
    qid = str(q["id"])
    section = str(q.get("section") or "")
    section_meta = SECTION_HEADERS.get(section, {})

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
            "source_exam": "PAES Historia y Ciencias Sociales",
            "source_form": SOURCE_FORM,
            "source_question_id": qid,
            "source_page": q.get("page"),
            "section": section,
            "subject": section_meta.get("subject"),
            "subject_es": section_meta.get("subject_es"),
            "regionality": section_meta.get("regionality", "regional"),
            "difficulty": "hard",
            "n_options": len(q["options"]),
            "option_letters": sorted(q["options"].keys()),
            "license_note": (
                "DEMRE/Universidad de Chile material; the question PDF states copyright and "
                "prohibits total or partial reproduction. Verify reuse permissions before redistribution."
            ),
        },
    }


def validate_questions_and_answers(
    questions: List[Dict[str, object]],
    answers: Dict[str, str],
) -> List[str]:
    warnings: List[str] = []

    if len(questions) != 45:
        warnings.append(f"Expected 45 DEMRE selected questions, found {len(questions)}.")

    qids = [str(q["id"]) for q in questions]
    duplicates = sorted({x for x in qids if qids.count(x) > 1}, key=lambda x: int(x))
    if duplicates:
        warnings.append(f"Duplicate question IDs: {duplicates}")

    missing = [qid for qid in qids if qid not in answers]
    if missing:
        warnings.append(f"Missing answers for parsed question IDs: {missing}")

    extra = sorted([qid for qid in answers if qid not in qids], key=lambda x: int(x))
    if extra:
        warnings.append(f"Answer key contains {len(extra)} IDs not present in parsed questions: {extra[:30]}")

    for q in questions:
        qid = str(q["id"])
        ans = answers.get(qid)
        if ans is None:
            continue
        options = q["options"]
        if ans not in options:
            warnings.append(f"Answer {ans} not present among options for question {qid}.")

    return warnings


def write_jsonl(records: Iterable[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default=None, help="Path to DEMRE question PDF.")
    parser.add_argument("--answer-pdf", default=None, help="Path to official DEMRE clavijero answer PDF.")
    parser.add_argument("--answers-json", default=None, help="Path to JSON mapping question IDs to answer letters.")
    parser.add_argument("--output", default=None, help="Output JSONL path.")
    parser.add_argument("--allow-extra-answer-ids", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    pdf_path = (
        Path(args.pdf)
        if args.pdf
        else ensure_raw_file("spanish_mcq/demre.pdf", download=not args.no_download)
    )
    answer_pdf = (
        Path(args.answer_pdf)
        if args.answer_pdf
        else ensure_raw_file("spanish_mcq/demre-answer-key.pdf", download=not args.no_download)
    )
    answers_json = Path(args.answers_json) if args.answers_json else None
    output_path = (
        Path(args.output)
        if args.output
        else get_processed_data_dir() / "demre" / "demre.jsonl"
    )

    lines = extract_lines(pdf_path)
    questions = parse_questions(lines)
    answers = load_answers(answer_pdf=answer_pdf, answers_json=answers_json)

    warnings = validate_questions_and_answers(questions, answers)

    blocking_warnings = []
    for w in warnings:
        if w.startswith("Missing answers"):
            blocking_warnings.append(w)
        if "not present among options" in w:
            blocking_warnings.append(w)
        if w.startswith("Answer key contains") and not args.allow_extra_answer_ids:
            blocking_warnings.append(w)

    if blocking_warnings:
        print(json.dumps({
            "status": "validation_failed",
            "pdf": str(pdf_path),
            "answer_pdf": str(answer_pdf) if answer_pdf else None,
            "answers_json": str(answers_json) if answers_json else None,
            "n_questions": len(questions),
            "n_answers": len(answers),
            "warnings": warnings,
            "blocking_warnings": blocking_warnings,
        }, ensure_ascii=False, indent=2))
        raise SystemExit(1)

    records: List[Dict[str, object]] = []
    for q in questions:
        qid = str(q["id"])
        answer = answers[qid]
        records.append(to_processed_record(q, answer, len(records)))

    write_jsonl(records, output_path)

    print(json.dumps({
        "status": "ok",
        "dataset": DATASET_NAME,
        "pdf": str(pdf_path),
        "answer_pdf": str(answer_pdf) if answer_pdf else None,
        "answers_json": str(answers_json) if answers_json else None,
        "output": str(output_path),
        "n_questions": len(questions),
        "n_answers": len(answers),
        "n_records": len(records),
        "warnings": warnings,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()