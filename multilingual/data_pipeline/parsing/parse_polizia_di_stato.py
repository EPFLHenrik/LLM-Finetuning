"""multilingual/data_pipeline/parsing/parse_polizia_di_stato.py

Parse the Italian `polizia-di-stato-banca-dati` PDF into processed MCQ JSONL.

The PDF contains numbered questions with options A-E, where option A is always
the correct answer in the original file. Because the PDF uses a two-column
layout and contains watermark noise, this script extracts text column-wise,
filters known noise patterns, shuffles the answer options using a fixed random
seed, updates the correct answer letter, and writes records in the standard
processed MCQ format used by the multilingual SFT pipeline.

Usage:
    python -m multilingual.data_pipeline.parsing.parse_polizia_di_stato \\
        --pdf multilingual/data/raw/italian_mcq_pdfs/2.polizia-di-stato-banca-dati.pdf \\
        --output multilingual/data/processed/polizia_di_stato/polizia_di_stato.jsonl \\
        --dataset polizia_di_stato \\
        --source polizia_di_stato \\
        --seed 42
"""

import re
import json
import random
import argparse
from pathlib import Path
import pdfplumber

WATERMARKS = {"TI", "DA", "NC", "BA"}
# Single letters that appear as watermarks in the right column
SINGLE_LETTER_RE = re.compile(r"^[A-Z]$")

SECTION_RE = re.compile(r"^(CULTURA GENERALE|MATEMATICA|LOGICA|INGLESE|INFORMATICA|DIRITTO(?:\s+\w+)?)$")
QUESTION_START_RE = re.compile(r"^(\d+)\.\s+(.+)$")
OPTION_RE = re.compile(r"^([A-E])\)\s+(.+)$")

SPLIT_X = 255  # column split x-coordinate (determined from word distribution)


def extract_column_texts(pdf_path: str) -> list[str]:
    """Extract text from each page as left-col + right-col concatenated."""
    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            left = page.crop((0, 0, SPLIT_X, page.height)).extract_text() or ""
            right = page.crop((SPLIT_X, 0, page.width, page.height)).extract_text() or ""
            texts.append(left + "\n" + right)
    return texts


def is_noise(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if s in WATERMARKS:
        return True
    # Single letters embedded as watermarks (but not option labels handled elsewhere)
    if SINGLE_LETTER_RE.match(s):
        return True
    # Page markers like "10 di 397"
    if re.match(r"^\d+ di \d+$", s):
        return True
    return False


def parse_questions(pages: list[str]) -> list[dict]:
    """State-machine parser over all page texts."""
    questions = []
    current_section = None
    q_num = None
    q_lines: list[str] = []
    opts: dict[str, str] = {}

    def flush():
        nonlocal q_num, q_lines, opts
        if q_num and q_lines and opts:
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
        lines = [l.strip() for l in page_text.splitlines()]
        i = 0
        while i < len(lines):
            line = lines[i]

            if is_noise(line):
                i += 1
                continue

            # Section header
            if SECTION_RE.match(line):
                flush()
                current_section = line
                i += 1
                continue

            # Option line: "A) text..."
            m = OPTION_RE.match(line)
            if m:
                letter, text = m.group(1), m.group(2).strip()
                j = i + 1
                while j < len(lines):
                    nxt = lines[j]
                    if not nxt or nxt in WATERMARKS or SINGLE_LETTER_RE.match(nxt) or \
                       OPTION_RE.match(nxt) or QUESTION_START_RE.match(nxt) or SECTION_RE.match(nxt):
                        break
                    text += " " + nxt
                    j += 1
                opts[letter] = text.strip()
                i = j
                continue

            # New question: "123. Question text..."
            m = QUESTION_START_RE.match(line)
            if m:
                flush()
                q_num = int(m.group(1))
                q_lines = [m.group(2).strip()]
                j = i + 1
                while j < len(lines):
                    nxt = lines[j]
                    if not nxt or nxt in WATERMARKS or SINGLE_LETTER_RE.match(nxt) or \
                       OPTION_RE.match(nxt) or QUESTION_START_RE.match(nxt) or SECTION_RE.match(nxt):
                        break
                    q_lines.append(nxt)
                    j += 1
                i = j
                continue

            i += 1

    flush()
    return questions


def shuffle_options(q: dict, rng: random.Random) -> dict:
    """Randomly reassign option letters, preserving correct answer content."""
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
    return {
        "id": f"{dataset}_italian_{idx}",
        "dataset": dataset,
        "language": "italian",
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
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", default="polizia_di_stato")
    parser.add_argument("--source", default="polizia_di_stato")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    print(f"Extracting text from {args.pdf}...")
    pages = extract_column_texts(args.pdf)

    print("Parsing questions...")
    questions = parse_questions(pages)
    print(f"  Found {len(questions)} questions")

    missing_a = [q for q in questions if "A" not in q["options"]]
    if missing_a:
        print(f"  WARNING: {len(missing_a)} questions missing option A — skipping")
        for q in missing_a[:3]:
            print(f"    Q{q['id']}: {q['question'][:60]} | opts={list(q['options'].keys())}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for q in questions:
            if len(q["options"]) < 2 or "A" not in q["options"]:
                continue
            q = shuffle_options(q, rng)
            record = to_sft_format(q, args.source, args.dataset, written)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Written {written} records to {out_path}")


if __name__ == "__main__":
    main()