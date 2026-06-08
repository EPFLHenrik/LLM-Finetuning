"""code/multilinguality_model/data_pipeline/parsing/parse_spain_boe_constitucion_espanola.py

Parse the BOE Spanish Constitution PDF into article-level JSONL.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pdfplumber

from multilinguality_model.paths import get_raw_data_dir
from multilinguality_model.data_pipeline.parsing.download_utils import ensure_raw_file


DATASET_NAME = "spain_boe_constitucion_espanola"
LAW_NAME = "Constitución Española"
COUNTRY = "Spain"
REGION = None
LANGUAGE = "spanish"
SOURCE_URL = "https://www.boe.es/buscar/pdf/1978/BOE-A-1978-31229-consolidado.pdf"
LICENSE = (
    "BOE public-sector information reuse terms / Ley 37/2007: reuse permitted subject "
    "to attribution to Agencia Estatal Boletín Oficial del Estado, non-distortion, "
    "non-implying official status, and indication that consolidated legislation is "
    "informative and has no legal value."
)

ARTICLE_RE = re.compile(r"^Artículo\s+([0-9]+(?:\s*bis|[A-Za-z])?)\.\s*(.*)$")
TITLE_RE = re.compile(r"^TÍTULO(?:\s+(?:PRELIMINAR|[IVXLCDM]+))?(?:\s*[.\-–]\s*(.*))?$", re.IGNORECASE)
CHAPTER_RE = re.compile(r"^CAPÍTULO\s+(?:[A-ZÁÉÍÓÚÑ]+|[IVXLCDM]+)(?:\s*[.\-–]\s*(.*))?$", re.IGNORECASE)
SECTION_RE = re.compile(r"^Sección\s+[0-9]+\.ª(?:\s+(.*))?$", re.IGNORECASE)
SUBSECTION_RE = re.compile(r"^Subsección\s+[0-9]+\.ª(?:\s+(.*))?$", re.IGNORECASE)
DISPOSITION_RE = re.compile(r"^Disposición\s+(adicional|transitoria|derogatoria|final)\b", re.IGNORECASE)

HEADER_FOOTER_EXACT = {
    "BOLETÍN OFICIAL DEL ESTADO",
    "LEGISLACIÓN CONSOLIDADA",
}
NOISE_PREFIXES = (
    "ÍNDICE",
)


def normalize_spaces(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def is_page_number_or_noise(line: str) -> bool:
    line = normalize_spaces(line)
    if not line:
        return True
    if line in HEADER_FOOTER_EXACT:
        return True
    if re.fullmatch(r"Página\s+\d+", line, flags=re.IGNORECASE):
        return True
    # Remove obvious index dot-leader lines from pages 1--2.
    if re.search(r"\.{4,}\s*\d+$", line):
        return True
    return False


def extract_clean_lines(pdf_path: str) -> List[str]:
    lines: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            text = page.extract_text(x_tolerance=1, y_tolerance=3) or ""
            for raw in text.splitlines():
                line = normalize_spaces(raw)
                if is_page_number_or_noise(line):
                    continue
                # Skip the table of contents pages; the actual text starts at page 3.
                if page_idx <= 2:
                    continue
                lines.append(line)
    return lines


def is_heading_or_boundary(line: str) -> bool:
    return bool(
        ARTICLE_RE.match(line)
        or TITLE_RE.match(line)
        or CHAPTER_RE.match(line)
        or SECTION_RE.match(line)
        or SUBSECTION_RE.match(line)
        or DISPOSITION_RE.match(line)
        or line in {"PREÁMBULO", "CONSTITUCIÓN", "TEXTO CONSOLIDADO"}
        or line.startswith("Última modificación:")
    )


def combine_heading(lines: List[str], i: int, kind: str) -> Tuple[str, int]:
    current = lines[i]
    if kind in {"titulo", "capitulo"}:
        # In this BOE PDF, title/chapter labels often occupy one line and the descriptive
        # heading occupies the immediately following line.
        if i + 1 < len(lines):
            nxt = lines[i + 1]
            if not is_heading_or_boundary(nxt):
                return f"{current}. {nxt}", i + 1
    return current, i


def join_article_lines(lines: List[str]) -> str:
    cleaned: List[str] = []
    for line in lines:
        line = normalize_spaces(line)
        if line:
            cleaned.append(line)
    if not cleaned:
        return ""

    out = ""
    for line in cleaned:
        if not out:
            out = line
            continue

        # Preserve separation before numbered paragraphs and lettered clauses.
        if re.match(r"^(\d+\.|[a-z]\))\s+", line):
            out += "\n" + line
        # Keep article marker visually separate from body.
        elif ARTICLE_RE.match(line):
            out += "\n" + line
        # Join ordinary wrapped legal prose.
        else:
            if out.endswith("-"):
                out = out[:-1] + line
            else:
                out += " " + line

    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def make_record(
    article_number: str,
    article_title: Optional[str],
    article_lines: List[str],
    hierarchy: Dict[str, Optional[str]],
) -> Dict[str, object]:
    source_id = f"{DATASET_NAME}_article_{article_number.replace(' ', '')}"
    source_text = join_article_lines(article_lines)

    record = {
        "id": source_id,
        "source_id": source_id,
        "dataset": DATASET_NAME,
        "language": LANGUAGE,
        "split": "raw",
        "task_type": "source_article",
        "country": COUNTRY,
        "region": REGION,
        "source_url": SOURCE_URL,
        "license": LICENSE,
        "source_unit_type": "article",
        "law_name": LAW_NAME,
        "título": hierarchy.get("título"),
        "capítulo": hierarchy.get("capítulo"),
        "sección": hierarchy.get("sección"),
        "subsección": hierarchy.get("subsección"),
        "article_number": article_number,
        "article_title": article_title,
        "source_text": source_text,
        "metadata": {
            "country": COUNTRY,
            "region": REGION,
            "law_name": LAW_NAME,
            "source_url": SOURCE_URL,
            "license": LICENSE,
            "source_unit_type": "article",
            "título": hierarchy.get("título"),
            "capítulo": hierarchy.get("capítulo"),
            "sección": hierarchy.get("sección"),
            "subsección": hierarchy.get("subsección"),
            "article_number": article_number,
            "article_title": article_title,
            "regionality": "region_agnostic",
            "source": "BOE legislación consolidada",
        },
    }
    return record


def parse_articles(lines: List[str]) -> Tuple[List[Dict[str, object]], List[Dict[str, str]]]:
    hierarchy: Dict[str, Optional[str]] = {
        "título": None,
        "capítulo": None,
        "sección": None,
        "subsección": None,
    }

    records: List[Dict[str, object]] = []
    skipped: List[Dict[str, str]] = []

    current_article_number: Optional[str] = None
    current_article_title: Optional[str] = None
    current_article_lines: List[str] = []
    current_article_hierarchy: Dict[str, Optional[str]] = hierarchy.copy()

    def flush_current(reason: str = "new boundary") -> None:
        nonlocal current_article_number, current_article_title, current_article_lines, current_article_hierarchy
        if current_article_number is None:
            return
        text = join_article_lines(current_article_lines)
        if text:
            records.append(
                make_record(
                    current_article_number,
                    current_article_title,
                    current_article_lines,
                    current_article_hierarchy,
                )
            )
        else:
            skipped.append({"unit": f"Artículo {current_article_number}", "reason": f"empty text at {reason}"})
        current_article_number = None
        current_article_title = None
        current_article_lines = []

    i = 0
    while i < len(lines):
        line = lines[i]

        m_article = ARTICLE_RE.match(line)
        if m_article:
            flush_current("next article")
            current_article_number = normalize_spaces(m_article.group(1))
            title = normalize_spaces(m_article.group(2))
            current_article_title = title if title else None
            current_article_hierarchy = hierarchy.copy()
            current_article_lines = [f"Artículo {current_article_number}."]
            if current_article_title:
                current_article_lines[0] += f" {current_article_title}"
            i += 1
            continue

        if DISPOSITION_RE.match(line):
            flush_current("disposition block")
            # Dispositions are not articles; they are intentionally excluded for this article-only parser.
            skipped.append({"unit": line, "reason": "non-article constitutional provision excluded"})
            i += 1
            continue

        if TITLE_RE.match(line):
            flush_current("title heading")
            heading, j = combine_heading(lines, i, "titulo")
            hierarchy["título"] = heading
            hierarchy["capítulo"] = None
            hierarchy["sección"] = None
            hierarchy["subsección"] = None
            i = j + 1
            continue

        if CHAPTER_RE.match(line):
            flush_current("chapter heading")
            heading, j = combine_heading(lines, i, "capitulo")
            hierarchy["capítulo"] = heading
            hierarchy["sección"] = None
            hierarchy["subsección"] = None
            i = j + 1
            continue

        if SECTION_RE.match(line):
            flush_current("section heading")
            hierarchy["sección"] = line
            hierarchy["subsección"] = None
            i += 1
            continue

        if SUBSECTION_RE.match(line):
            flush_current("subsection heading")
            hierarchy["subsección"] = line
            i += 1
            continue

        # Preamble and royal formulae are intentionally ignored until the first article.
        if current_article_number is not None:
            current_article_lines.append(line)

        i += 1

    flush_current("end of file")
    return records, skipped


def validate(records: List[Dict[str, object]]) -> List[str]:
    warnings: List[str] = []
    nums = [str(r["article_number"]) for r in records]
    expected = [str(i) for i in range(1, 170)]
    missing = [n for n in expected if n not in nums]
    duplicates = sorted({n for n in nums if nums.count(n) > 1}, key=lambda x: int(re.sub(r"\D", "", x) or 0))
    if missing:
        warnings.append(f"Missing expected articles: {missing}")
    if duplicates:
        warnings.append(f"Duplicate article numbers: {duplicates}")
    for r in records:
        if not str(r.get("source_text", "")).strip():
            warnings.append(f"Empty source_text: {r.get('source_id')}")
    return warnings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default=None, help="Path to the BOE consolidated Spanish Constitution PDF.")
    parser.add_argument("--output", default=None, help="Output JSONL path.")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    pdf_path = (
        Path(args.pdf)
        if args.pdf
        else ensure_raw_file(
            "spain_boe/spain_boe_constitucion_espanola_consolidada.pdf",
            download=not args.no_download,
        )
    )
    output_path = (
        Path(args.output)
        if args.output
        else get_raw_data_dir() / "spain_boe_legislation" / "spain_boe_constitucion_espanola.jsonl"
    )

    lines = extract_clean_lines(str(pdf_path))
    records, skipped = parse_articles(lines)
    warnings = validate(records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(json.dumps({
        "dataset": DATASET_NAME,
        "pdf": str(pdf_path),
        "output": str(output_path),
        "n_records": len(records),
        "n_skipped_non_article_units": len(skipped),
        "warnings": warnings,
        "skipped_preview": skipped[:20],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
