"""preprocess_mexican_civil_code.py

Downloads the Mexican Federal Civil Code PDF from diputados.gob.mx,
extracts and parses articles into JSONL format compatible with
generate_italian_civil_code_mcq.py (same schema).

Output JSONL schema:
  article_id:    "art1", "art2", ...
  article_title: "" (Mexican code has no per-article titles, left empty)
  article_text:  full article text
  article_references: null

Usage:
    python preprocess_mexican_civil_code.py \
        --output_dir multilingual/data/processed/mexican_civil_code
"""

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path


PDF_URL  = "http://www.diputados.gob.mx/LeyesBiblio/pdf/CCF.pdf"
MIN_CHARS = 80  # skip articles shorter than this


def download_pdf(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"PDF already exists: {dest}")
        return
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved to {dest}")


def extract_text_from_pdf(pdf_path: Path) -> str:
    try:
        import pdfplumber
        print("Extracting text with pdfplumber ...")
        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return "\n".join(pages)
    except ImportError:
        pass

    try:
        import pypdf
        print("Extracting text with pypdf ...")
        reader = pypdf.PdfReader(str(pdf_path))
        return "\n".join(
            page.extract_text() or "" for page in reader.pages
        )
    except ImportError:
        pass

    # Fallback: pdftotext CLI
    import subprocess
    print("Extracting text with pdftotext ...")
    result = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pdftotext failed: {result.stderr}")
    return result.stdout


def _clean_text(text: str) -> str:
    """Normalize whitespace and remove PDF artifacts."""
    # Join hyphenated line breaks
    text = re.sub(r"-\n(\w)", r"\1", text)
    # Collapse multiple spaces
    text = re.sub(r"[ \t]+", " ", text)
    # Remove page headers/footers (common patterns in Mexican law PDFs)
    text = re.sub(r"CÁMARA DE DIPUTADOS.*?\n", "", text)
    text = re.sub(r"Secretaría General.*?\n", "", text)
    text = re.sub(r"Última Reforma.*?\n", "", text)
    text = re.sub(r"\f", "\n", text)  # form feeds
    return text


def parse_articles(raw_text: str) -> list[dict]:
    """
    Parse articles from raw PDF text.
    Mexican Civil Code format: 'Artículo N.- text'
    Also handles 'Artículo No.' and 'Artículo N°'
    """
    text = _clean_text(raw_text)

    # Split on article boundaries
    # Matches: Artículo 1.- / Artículo 1o.- / Artículo 1°.-
    pattern = re.compile(
        r"(?:^|\n)\s*(Art[íi]culo\s+(\d+)[°ºo]?\s*[\.\-–]+)",
        re.IGNORECASE | re.MULTILINE,
    )

    matches = list(pattern.finditer(text))
    print(f"Found {len(matches)} article markers")

    articles = []
    for i, match in enumerate(matches):
        art_num = match.group(2)
        start   = match.end()
        end     = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        art_text = text[start:end].strip()

        # Remove amendment notes like "Artículo reformado DOF..."
        art_text = re.sub(r"\n?\s*Art[íi]culo\s+reformado\s+DOF[^\n]*", "", art_text)
        art_text = re.sub(r"\n?\s*P[áa]rrafo\s+reformado\s+DOF[^\n]*", "", art_text)
        art_text = re.sub(r"\n?\s*Fracci[óo]n\s+\w+\s+reformada\s+DOF[^\n]*", "", art_text)
        art_text = re.sub(r"\n?\s*Art[íi]culo\s+derogado\s+DOF[^\n]*", "", art_text)
        art_text = re.sub(r"\n?\s*Fe de erratas[^\n]*", "", art_text)
        art_text = re.sub(r"\s+", " ", art_text).strip()

        if len(art_text) < MIN_CHARS:
            continue

        # Skip derogated (abrogated) articles
        if re.match(r"\(derogado\)", art_text, re.IGNORECASE):
            continue

        articles.append({
            "article_id":         f"art{art_num}",
            "article_title":      "",  # Mexican code has no per-article titles
            "article_text":       art_text.lower(),  # lowercase like Italian dataset
            "article_references": None,
        })

    return articles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path,
                        default=Path("multilingual/data/processed/mexican_civil_code"))
    parser.add_argument("--pdf_path", type=Path, default=None,
                        help="Path to already-downloaded PDF (skip download).")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.pdf_path or (args.output_dir / "CCF.pdf")

    # Download
    download_pdf(PDF_URL, pdf_path)

    # Extract
    raw_text = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(raw_text):,} characters")

    # Parse
    articles = parse_articles(raw_text)
    print(f"Parsed {len(articles)} articles (≥{MIN_CHARS} chars)")

    # Write JSONL
    out_path = args.output_dir / "mexican_civil_code_all.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for art in articles:
            f.write(json.dumps(art, ensure_ascii=False) + "\n")

    print(f"Saved {len(articles)} articles → {out_path}")

    # Print a few examples
    print("\n--- Sample articles ---")
    for art in articles[:3]:
        print(f"[{art['article_id']}] {art['article_text'][:150]}...")
        print()


if __name__ == "__main__":
    main()