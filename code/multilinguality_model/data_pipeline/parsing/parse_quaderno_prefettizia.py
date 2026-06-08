"""code/multilinguality_model/data_pipeline/parsing/parse_quaderno_prefettizia.py

Parse the Italian Quaderno Prefettizia MCQ PDF into processed JSONL.
"""

import re
import json
import random
import argparse
from pathlib import Path
import pdfplumber

from multilinguality_model.paths import get_processed_data_dir
from multilinguality_model.data_pipeline.parsing.download_utils import ensure_raw_file

# Section headers like "DIRITTO COMUNITARIO - DIFFICOLTA' FACILE"
SECTION_RE = re.compile(r"^(.+?)\s*-\s*DIFFICOLT[AÀ]'?\s*(FACILE|MEDIA|ALTA)$")
PAGE_RE = re.compile(r"^Pagina \d+$")
QUESTION_NUM_RE = re.compile(r"^\d+$")
OPTION_RE = re.compile(r"^([A-D])\s+(.+)$")


def extract_text_pages(pdf_path: str) -> list[str]:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return pages


def parse_questions(pages: list[str]) -> list[dict]:
    """Parse all pages into list of MCQ dicts."""
    questions = []
    current_section = None

    # State machine
    q_num = None
    q_lines = []
    opts = {}

    def flush_question():
        nonlocal q_num, q_lines, opts
        if q_num and q_lines and len(opts) >= 2:
            questions.append({
                "id": q_num,
                "section": current_section,
                "question": " ".join(q_lines).strip(),
                "options": opts.copy(),
                "correct": "A",
            })
        q_num = None
        q_lines = []
        opts = {}

    for page_text in pages:
        lines = page_text.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Skip empty, page markers, form feeds
            if not line or PAGE_RE.match(line) or line == "\f":
                i += 1
                continue

            # Section header
            m = SECTION_RE.match(line)
            if m:
                flush_question()
                current_section = f"{m.group(1).strip()} - DIFFICOLTA' {m.group(2)}"
                i += 1
                continue

            # Option line (A ... D)
            m = OPTION_RE.match(line)
            if m:
                letter, text = m.group(1), m.group(2).strip()
                # Accumulate multi-line options
                j = i + 1
                while j < len(lines):
                    nxt = lines[j].strip()
                    if not nxt or OPTION_RE.match(nxt) or QUESTION_NUM_RE.match(nxt):
                        break
                    text += " " + nxt
                    j += 1
                opts[letter] = text.strip()
                i = j
                continue

            # Standalone number = question number
            if QUESTION_NUM_RE.match(line):
                flush_question()
                q_num = int(line)
                i += 1
                continue

            # Otherwise: question body text
            if q_num is not None and not opts:
                q_lines.append(line)

            i += 1

    flush_question()
    return questions


def shuffle_options(q: dict, rng: random.Random) -> dict:
    """Randomly reassign option letters, track new correct letter."""
    letters = sorted(q["options"].keys())
    texts = [q["options"][l] for l in letters]
    correct_text = q["options"][q["correct"]]

    rng.shuffle(texts)
    new_opts = {l: t for l, t in zip(letters, texts)}
    new_correct = next(l for l, t in new_opts.items() if t == correct_text)

    return {**q, "options": new_opts, "correct": new_correct}


def to_sft_format(q: dict, source: str, dataset: str, idx: int) -> dict:
    """Convert to standard processed JSONL format matching existing pipeline."""
    opts_str = "\n".join(f"{l}) {t}" for l, t in sorted(q["options"].items()))
    record_id = f"{dataset}_italian_{idx}"
    return {
        "id": record_id,
        "source_id": record_id,
        "dataset": dataset,
        "language": "italian",
        "split": "raw",
        "task_type": "mcq",
        "prompt": f"{q['question']}\n\n{opts_str}",
        "answer": q["correct"],
        "metadata": {
            "source": source,
            "section": q.get("section", ""),
            "original_id": q.get("id", ""),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default=None, help="Path to PDF")
    parser.add_argument("--output", default=None, help="Output JSONL path")
    parser.add_argument("--dataset", default="quaderno_prefettizia")
    parser.add_argument("--source", default="quaderno_prefettizia", help="Source tag")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    pdf_path = (
        Path(args.pdf)
        if args.pdf
        else ensure_raw_file("italian_mcq/quaderno-delle-domande.pdf", download=not args.no_download)
    )
    output_path = (
        Path(args.output)
        if args.output
        else get_processed_data_dir() / "quaderno_prefettizia" / "quaderno_prefettizia_initial_parsed.jsonl"
    )

    rng = random.Random(args.seed)
    print(f"Extracting text from {pdf_path}...")
    pages = extract_text_pages(str(pdf_path))

    print("Parsing questions...")
    questions = parse_questions(pages)
    print(f"  Found {len(questions)} questions")

    out_path = output_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for q in questions:
            if len(q["options"]) < 2:
                continue
            q = shuffle_options(q, rng)
            record = to_sft_format(q, args.source, args.dataset, written)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Written {written} records to {out_path}")


if __name__ == "__main__":
    main()