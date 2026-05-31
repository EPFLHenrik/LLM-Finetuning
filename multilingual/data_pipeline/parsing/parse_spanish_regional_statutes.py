"""multilingual/data_pipeline/parsing/parse_spanish_regional_statutes.py

Parse BOE consolidated PDFs of selected Spanish Autonomous Community Statutes
into article-level JSONL files.

The script reads the regional statute PDFs listed in `STATUTES`, removes common
PDF noise such as headers, footers, page numbers, and index lines, then extracts
regular `Artículo N.` units with their legal hierarchy metadata.

By default, PDFs are read from:

    multilingual/data/raw/spanish-extra

and outputs are written to:

    multilingual/data/raw/spain_boe_legislation/<dataset_name>.jsonl

Usage:
    python -m multilingual.data_pipeline.parsing.parse_spanish_regional_statutes \\
        --input-dir multilingual/data/raw/spanish-extra \\
        --output-root multilingual/data/raw

Dry run:
    python -m multilingual.data_pipeline.parsing.parse_spanish_regional_statutes \\
        --input-dir multilingual/data/raw/spanish-extra \\
        --output-root multilingual/data/raw \\
        --dry-run

Parse selected statutes only:
    python -m multilingual.data_pipeline.parsing.parse_spanish_regional_statutes \\
        --input-dir multilingual/data/raw/spanish-extra \\
        --output-root multilingual/data/raw \\
        --only cataluna galicia madrid
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import pdfplumber
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: pdfplumber. Install it with `pip install pdfplumber`."
    ) from exc


LICENSE = (
    "BOE public-sector information reuse terms / Ley 37/2007: reuse permitted subject "
    "to attribution to Agencia Estatal Boletín Oficial del Estado, non-distortion, "
    "non-implying official status, and indication that consolidated legislation is "
    "informative and has no legal value."
)


@dataclass(frozen=True)
class StatuteConfig:
    dataset: str
    filename: str
    law_name: str
    region: str
    boe_id: str

    @property
    def source_url(self) -> str:
        year = self.boe_id.split("-")[2]
        return f"https://www.boe.es/buscar/pdf/{year}/{self.boe_id}-consolidado.pdf"


STATUTES: Dict[str, StatuteConfig] = {
    "andalucia": StatuteConfig(
        dataset="spain_boe_estatuto_andalucia",
        filename="spain_boe_estatuto_andalucia_consolidado.pdf",
        law_name="Ley Orgánica 2/2007, de 19 de marzo, de reforma del Estatuto de Autonomía para Andalucía",
        region="Andalucía",
        boe_id="BOE-A-2007-5825",
    ),
    "canarias": StatuteConfig(
        dataset="spain_boe_estatuto_canarias",
        filename="spain_boe_estatuto_canarias_consolidado.pdf",
        law_name="Ley Orgánica 1/2018, de 5 de noviembre, de reforma del Estatuto de Autonomía de Canarias",
        region="Canarias",
        boe_id="BOE-A-2018-15138",
    ),
    "cataluna": StatuteConfig(
        dataset="spain_boe_estatuto_cataluna",
        filename="spain_boe_estatuto_cataluna_consolidado.pdf",
        law_name="Ley Orgánica 6/2006, de 19 de julio, de reforma del Estatuto de Autonomía de Cataluña",
        region="Cataluña",
        boe_id="BOE-A-2006-13087",
    ),
    "comunitat_valenciana": StatuteConfig(
        dataset="spain_boe_estatuto_comunitat_valenciana",
        filename="spain_boe_estatuto_comunitat_valenciana_consolidado.pdf",
        law_name="Ley Orgánica 5/1982, de 1 de julio, de Estatuto de Autonomía de la Comunidad Valenciana",
        region="Comunitat Valenciana",
        boe_id="BOE-A-1982-17235",
    ),
    "galicia": StatuteConfig(
        dataset="spain_boe_estatuto_galicia",
        filename="spain_boe_estatuto_galicia_consolidado.pdf",
        law_name="Ley Orgánica 1/1981, de 6 de abril, de Estatuto de Autonomía para Galicia",
        region="Galicia",
        boe_id="BOE-A-1981-9564",
    ),
    "madrid": StatuteConfig(
        dataset="spain_boe_estatuto_madrid",
        filename="spain_boe_estatuto_madrid_consolidado.pdf",
        law_name="Ley Orgánica 3/1983, de 25 de febrero, de Estatuto de Autonomía de la Comunidad de Madrid",
        region="Madrid",
        boe_id="BOE-A-1983-6317",
    ),
    "pais_vasco": StatuteConfig(
        dataset="spain_boe_estatuto_pais_vasco",
        filename="spain_boe_estatuto_pais_vasco_consolidado.pdf",
        law_name="Ley Orgánica 3/1979, de 18 de diciembre, de Estatuto de Autonomía para el País Vasco",
        region="País Vasco",
        boe_id="BOE-A-1979-30177",
    ),
}


ARTICLE_RE = re.compile(
    r"^Artículo\s+([0-9]+(?:\s*[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+)?(?:\.[0-9]+)?)\.?(?:\s+(.*))?$"
)
TITLE_RE = re.compile(r"^T[ÍI]TULO\s+(?:PRELIMINAR|[IVXLCDM]+|\d+)(?:\b|\.)\s*(.*)$", re.IGNORECASE)
CHAPTER_RE = re.compile(r"^CAP[ÍI]TULO\s+(?:PRELIMINAR|[IVXLCDM]+|\d+|PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO|SEXTO|S[ÉE]PTIMO|OCTAVO|NOVENO|D[ÉE]CIMO)(?:\b|\.)\s*(.*)$", re.IGNORECASE)
SECTION_RE = re.compile(r"^Secci[oó]n\s+(?:[a-záéíóúüñ]+|[0-9]+\.?[ªº]?)(?:\b|\.)\s*(.*)$", re.IGNORECASE)
SUBSECTION_RE = re.compile(r"^Subsecci[oó]n\s+(?:[a-záéíóúüñ]+|[0-9]+\.?[ªº]?)(?:\b|\.)\s*(.*)$", re.IGNORECASE)
PROVISION_RE = re.compile(r"^(Disposici[oó]n|Disposiciones|ANEXO\b|Anexo\b)", re.IGNORECASE)
DOT_LEADER_RE = re.compile(r"\.{3,}|(?:\.\s*){3,}")
PAGE_RE = re.compile(r"^P[áa]gina\s+\d+$", re.IGNORECASE)

ROMAN_OR_ORDINAL_TITLE_ONLY_RE = re.compile(
    r"^(T[ÍI]TULO\s+(?:PRELIMINAR|[IVXLCDM]+|\d+)|CAP[ÍI]TULO\s+(?:PRELIMINAR|[IVXLCDM]+|\d+|PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO|SEXTO|S[ÉE]PTIMO|OCTAVO|NOVENO|D[ÉE]CIMO)|Secci[oó]n\s+(?:[a-záéíóúüñ]+|[0-9]+\.?[ªº]?)|Subsecci[oó]n\s+(?:[a-záéíóúüñ]+|[0-9]+\.?[ªº]?))\.?$",
    re.IGNORECASE,
)

PARAGRAPH_START_RE = re.compile(
    r"^(?:\d+\.|[a-záéíóúüñ]\)|[0-9]+\.[0-9]+\b|Uno\.|Dos\.|Tres\.|Cuatro\.|Cinco\.|Seis\.|Siete\.|Ocho\.|Nueve\.|Diez\.|Téngase\s+en\s+cuenta|V[ée]ase|Ref\.)",
    re.IGNORECASE,
)


def clean_line(line: str) -> str:
    line = line.replace("\ufeff", "").replace("\ufffe", "")
    line = line.replace("\u00ad", "")
    line = re.sub(r"\s+", " ", line).strip()
    return line


def is_noise_line(line: str) -> bool:
    if not line:
        return True
    upper = line.upper()
    if upper in {"BOLETÍN OFICIAL DEL ESTADO", "LEGISLACIÓN CONSOLIDADA"}:
        return True
    if PAGE_RE.match(line):
        return True
    return False


def is_index_line(line: str) -> bool:
    if line.upper() == "ÍNDICE":
        return True
    if DOT_LEADER_RE.search(line):
        return True
    return False


def extract_clean_lines(pdf_path: Path) -> Tuple[List[str], str]:
    lines: List[str] = []
    first_pages_text: List[str] = []
    in_consolidated_text = False

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            text = page.extract_text(x_tolerance=1.5, y_tolerance=3) or ""
            if page_idx < 2:
                first_pages_text.append(text)
            for raw in text.splitlines():
                line = clean_line(raw)
                if is_noise_line(line) or is_index_line(line):
                    continue
                if line.upper() == "TEXTO CONSOLIDADO":
                    in_consolidated_text = True
                    continue
                if not in_consolidated_text:
                    continue
                if re.match(r"^Última modificación:", line, flags=re.IGNORECASE):
                    continue
                lines.append(line)

    return lines, "\n".join(first_pages_text)


def heading_level(line: str) -> Optional[str]:
    if TITLE_RE.match(line):
        return "título"
    if CHAPTER_RE.match(line):
        return "capítulo"
    if SECTION_RE.match(line):
        return "sección"
    if SUBSECTION_RE.match(line):
        return "subsección"
    return None


def is_bare_heading(line: str) -> bool:
    return bool(ROMAN_OR_ORDINAL_TITLE_ONLY_RE.match(line))


def normalize_heading(label: str, descriptor: Optional[str]) -> str:
    label = clean_line(label).rstrip(".")
    descriptor = clean_line(descriptor or "")
    if descriptor:
        return f"{label}. {descriptor}"
    return label


def update_context(context: Dict[str, Optional[str]], level: str, value: str) -> None:
    context[level] = value
    if level == "título":
        context["capítulo"] = None
        context["sección"] = None
        context["subsección"] = None
    elif level == "capítulo":
        context["sección"] = None
        context["subsección"] = None
    elif level == "sección":
        context["subsección"] = None


def normalize_article_text(lines: List[str]) -> str:
    paragraphs: List[str] = []
    for idx, line in enumerate(lines):
        line = clean_line(line)
        if not line:
            continue
        if not paragraphs:
            paragraphs.append(line)
            continue
        if idx == 1 or PARAGRAPH_START_RE.match(line):
            paragraphs.append(line)
        elif paragraphs[-1].endswith("-"):
            paragraphs[-1] = paragraphs[-1][:-1] + line
        else:
            paragraphs[-1] = paragraphs[-1] + " " + line
    text = "\n".join(paragraphs)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def make_record(
    cfg: StatuteConfig,
    article_number: str,
    article_title: Optional[str],
    article_lines: List[str],
    context: Dict[str, Optional[str]],
) -> Dict[str, object]:
    clean_number = re.sub(r"\s+", "", article_number.strip())
    record_id = f"{cfg.dataset}_article_{clean_number}"
    source_text = normalize_article_text(article_lines)

    metadata = {
        "country": "Spain",
        "region": cfg.region,
        "law_name": cfg.law_name,
        "source_url": cfg.source_url,
        "license": LICENSE,
        "source_unit_type": "article",
        "título": context.get("título"),
        "capítulo": context.get("capítulo"),
        "sección": context.get("sección"),
        "subsección": context.get("subsección"),
        "article_number": clean_number,
        "article_title": article_title,
        "regionality": "regional_explicit",
        "source": "BOE legislación consolidada",
        "boe_id": cfg.boe_id,
    }

    return {
        "id": record_id,
        "source_id": record_id,
        "dataset": cfg.dataset,
        "language": "spanish",
        "split": "raw",
        "task_type": "source_article",
        "country": "Spain",
        "region": cfg.region,
        "source_url": cfg.source_url,
        "license": LICENSE,
        "source_unit_type": "article",
        "law_name": cfg.law_name,
        "título": context.get("título"),
        "capítulo": context.get("capítulo"),
        "sección": context.get("sección"),
        "subsección": context.get("subsección"),
        "article_number": clean_number,
        "article_title": article_title,
        "source_text": source_text,
        "metadata": metadata,
    }


def parse_statute(pdf_path: Path, cfg: StatuteConfig) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    lines, first_pages_text = extract_clean_lines(pdf_path)
    header_text = first_pages_text + "\n" + "\n".join(lines[:30])
    header_ok = cfg.boe_id in header_text

    records: List[Dict[str, object]] = []
    skipped: List[Dict[str, str]] = []
    warnings: List[str] = []
    if not header_ok:
        warnings.append(
            f"Expected BOE id {cfg.boe_id} was not found in the PDF header/body preview. "
            "The file may be misnamed or may not correspond to the registry entry."
        )

    context: Dict[str, Optional[str]] = {
        "título": None,
        "capítulo": None,
        "sección": None,
        "subsección": None,
    }
    pending_heading: Optional[Tuple[str, str]] = None
    current_article: Optional[Dict[str, object]] = None
    current_lines: List[str] = []

    def flush_pending_heading() -> None:
        nonlocal pending_heading
        if pending_heading is not None:
            level, value = pending_heading
            update_context(context, level, value)
            pending_heading = None

    def finish_article() -> None:
        nonlocal current_article, current_lines
        if current_article is None:
            return
        rec = make_record(
            cfg=cfg,
            article_number=str(current_article["number"]),
            article_title=current_article.get("title"),
            article_lines=current_lines,
            context=current_article["context"],
        )
        if rec["source_text"]:
            records.append(rec)
        else:
            skipped.append({"unit": f"Artículo {current_article['number']}.", "reason": "empty source_text after cleaning"})
        current_article = None
        current_lines = []

    for line in lines:
        if not line:
            continue
        if line.upper() in {"PREÁMBULO", "PREAMBULO"}:
            continue

        article_match = ARTICLE_RE.match(line)
        if article_match:
            flush_pending_heading()
            finish_article()
            number = clean_line(article_match.group(1))
            title = clean_line(article_match.group(2) or "") or None
            heading = f"Artículo {number}." + (f" {title}" if title else "")
            current_article = {
                "number": number,
                "title": title,
                "context": dict(context),
            }
            current_lines = [heading]
            continue

        level = heading_level(line)
        if level:
            finish_article()
            flush_pending_heading()
            if is_bare_heading(line):
                pending_heading = (level, clean_line(line).rstrip("."))
            else:
                update_context(context, level, clean_line(line).rstrip("."))
            continue

        if pending_heading is not None:
            if not ARTICLE_RE.match(line) and not heading_level(line) and not PROVISION_RE.match(line):
                level, value = pending_heading
                pending_heading = (level, normalize_heading(value, line))
                continue
            flush_pending_heading()

        if PROVISION_RE.match(line):
            finish_article()
            skipped.append({"unit": line, "reason": "non-article statutory provision or annex excluded"})
            continue

        if current_article is not None:
            current_lines.append(line)

    flush_pending_heading()
    finish_article()

    article_numbers = [r["article_number"] for r in records]
    duplicate_numbers = sorted({n for n in article_numbers if article_numbers.count(n) > 1})
    if duplicate_numbers:
        warnings.append(f"Duplicate article numbers found: {duplicate_numbers}")

    summary = {
        "dataset": cfg.dataset,
        "pdf": str(pdf_path),
        "n_records": len(records),
        "n_skipped_non_article_units": len(skipped),
        "warnings": warnings,
        "skipped_preview": skipped[:30],
    }
    return records, summary


def write_jsonl(records: Iterable[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def default_repo_root() -> Path:
    # Script is expected at multilingual/distillation/parse_spanish_regional_statutes.py
    return Path(__file__).resolve().parents[2]


def resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    repo_root = default_repo_root()
    input_dir = Path(args.input_dir) if args.input_dir else repo_root / "multilingual" / "data" / "raw" / "spanish-extra"
    output_root = Path(args.output_root) if args.output_root else repo_root / "multilingual" / "data" / "raw"
    return input_dir.resolve(), output_root.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse BOE regional autonomy statutes into article-level JSONL.")
    parser.add_argument("--input-dir", type=str, default=None, help="Directory containing the BOE statute PDFs.")
    parser.add_argument("--output-root", type=str, default=None, help="Root directory for output dataset folders.")
    parser.add_argument(
        "--only",
        nargs="*",
        choices=sorted(STATUTES.keys()),
        help="Optional subset to parse, e.g. --only cataluna galicia madrid.",
    )
    parser.add_argument(
        "--allow-header-mismatch",
        action="store_true",
        help="Allow parsing even if the expected BOE id is not found in the PDF preview.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Parse and print summaries without writing JSONL files.")
    args = parser.parse_args()

    input_dir, output_root = resolve_paths(args)
    selected_keys = args.only or sorted(STATUTES.keys())

    all_summaries: List[Dict[str, object]] = []
    failures = 0

    for key in selected_keys:
        cfg = STATUTES[key]
        pdf_path = input_dir / cfg.filename
        output_path = output_root / "spain_boe_legislation" / f"{cfg.dataset}.jsonl"

        if not pdf_path.exists():
            summary = {
                "dataset": cfg.dataset,
                "pdf": str(pdf_path),
                "output": str(output_path),
                "status": "missing_pdf",
            }
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            all_summaries.append(summary)
            failures += 1
            continue

        records, summary = parse_statute(pdf_path, cfg)
        summary["output"] = str(output_path)

        if summary["warnings"] and not args.allow_header_mismatch:
            summary["status"] = "skipped_due_to_header_mismatch"
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            all_summaries.append(summary)
            failures += 1
            continue

        if not args.dry_run:
            write_jsonl(records, output_path)

        summary["status"] = "written" if not args.dry_run else "dry_run_ok"
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        all_summaries.append(summary)

    summary_path = output_root / "spain_boe_legislation" / "regional_statutes_parse_summary.json"
    if not args.dry_run:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote summary: {summary_path}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
