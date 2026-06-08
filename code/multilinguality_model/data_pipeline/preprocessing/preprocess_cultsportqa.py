"""Preprocess CultSportQA into clean multilingual MCQA JSONL files."""

from io import BytesIO
from urllib.request import urlopen
import ast
import json
import re

import numpy as np
import pandas as pd
from multilinguality_model.paths import get_processed_data_dir



OUTPUT_DIR = get_processed_data_dir() / "cultsportqa"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/M-Groot7/CultSportQA/main"
COUNTRIES = ["China", "India"]
GROUPS = ["HBQ", "RBQ", "SBQ"]
VALID_ANSWERS = {"A", "B", "C", "D"}
LETTER_MAP = {
    "A": "A",
    "B": "B",
    "C": "C",
    "D": "D",
    "a": "A",
    "b": "B",
    "c": "C",
    "d": "D",
    "ए": "A",
    "बी": "B",
    "सी": "C",
    "डी": "D",
    "अ": "A",
    "ब": "B",
    "स": "C",
    "द": "D",
}

ANSWER_LABEL_RE = re.compile(
    r"^\s*(A|B|C|D|a|b|c|d|ए|बी|सी|डी|अ|ब|स|द)\s*[\)\]）\.:\-]?",
    re.UNICODE,
)

OPTION_LABEL_RE = re.compile(
    r"(A|B|C|D|a|b|c|d|ए|बी|सी|डी|अ|ब|स|द)\s*[\)\]）\.:\-]",
    re.UNICODE,
)

COUNTRY_TO_LANGUAGE = {
    "China": "chinese",
    "India": "hindi",
}

GROUP_DESCRIPTION = {
    "HBQ": "historical_background",
    "RBQ": "rules_reference",
    "SBQ": "scenario_based",
}


def normalize_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_for_dedup(x) -> str:
    x = normalize_text(x).lower()
    x = re.sub(r"\s+", " ", x)
    return x.strip()


def is_placeholder(x) -> bool:
    s = normalize_text(x).lower()
    return s in {
        "",
        "nan",
        "none",
        "error",
        "no correction needed",
        "no answer needed",
    }


def has_corruption(*fields) -> bool:
    return any("�" in normalize_text(x) for x in fields)


def parse_answer_label(answer):
    s = normalize_text(answer)
    if is_placeholder(s):
        return None

    match = ANSWER_LABEL_RE.match(s)
    if not match:
        return None

    return LETTER_MAP.get(match.group(1))


def parse_options(option_text):
    s = normalize_text(option_text)
    if is_placeholder(s):
        return None

    if s.startswith("[") and s.endswith("]"):
        try:
            parts = ast.literal_eval(s)
        except Exception:
            return None

        if not isinstance(parts, list):
            return None

        parts = [normalize_text(part) for part in parts]
    else:
        matches = list(OPTION_LABEL_RE.finditer(s))
        if not matches:
            return None

        parts = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(s)
            parts.append(s[start:end].strip())

    parsed = {}

    for part in parts:
        match = ANSWER_LABEL_RE.match(part)
        if not match:
            return None

        label = LETTER_MAP.get(match.group(1))
        if label is None:
            return None

        text = part[match.end():].strip()
        text = re.sub(r"^[\)\]）\.:\-\s、,，]+", "", text).strip()
        text = text.rstrip("、,，;；").strip()

        if not text:
            return None
        if label in parsed:
            return None

        parsed[label] = text

    if set(parsed.keys()) != {"A", "B", "C", "D"}:
        return None

    return parsed


def iter_source_files():
    for country in COUNTRIES:
        for group in GROUPS:
            filename = f"{country}_{group}.xlsx"
            url = f"{GITHUB_RAW_BASE}/{country}/{group}/{filename}"
            yield country, group, filename, url


def load_excel_from_url(url: str) -> pd.ExcelFile:
    with urlopen(url, timeout=60) as response:
        content = response.read()
    return pd.ExcelFile(BytesIO(content), engine="openpyxl")


def load_raw_dataframe():
    raw_parts = []

    for country, group, filename, url in iter_source_files():
        print(f"Loading {url}")

        xls = load_excel_from_url(url)

        for sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet, engine="openpyxl")
            df.columns = [str(col).strip().lower() for col in df.columns]

            expected_cols = ["question", "option", "answer"]
            missing = [col for col in expected_cols if col not in df.columns]
            if missing:
                continue

            df = df[expected_cols].copy()
            df.insert(0, "country", country)
            df.insert(1, "group", group)
            df.insert(2, "sport", sheet)
            df.insert(3, "row_idx", range(len(df)))
            df.insert(4, "source_file", filename)
            df.insert(5, "source_url", url)

            raw_parts.append(df)

    if not raw_parts:
        return pd.DataFrame(
            columns=[
                "country",
                "group",
                "sport",
                "row_idx",
                "source_file",
                "source_url",
                "question",
                "option",
                "answer",
            ]
        )

    return pd.concat(raw_parts, ignore_index=True)


def build_clean_dataframe(raw_df):
    clean_rows = []
    rejected_rows = []

    for _, row in raw_df.iterrows():
        country = normalize_text(row.get("country"))
        group = normalize_text(row.get("group"))
        sport = normalize_text(row.get("sport"))
        row_idx = row.get("row_idx")
        source_file = normalize_text(row.get("source_file"))
        source_url = normalize_text(row.get("source_url"))

        question = normalize_text(row.get("question"))
        option_raw = normalize_text(row.get("option"))
        answer_raw = normalize_text(row.get("answer"))

        reject_reason = None

        if country not in COUNTRY_TO_LANGUAGE:
            reject_reason = "unsupported_country"
        elif is_placeholder(question):
            reject_reason = "invalid_question"
        elif is_placeholder(option_raw):
            reject_reason = "invalid_options"
        elif is_placeholder(answer_raw):
            reject_reason = "invalid_answer"
        elif has_corruption(question, option_raw, answer_raw):
            reject_reason = "corruption"
        else:
            options = parse_options(option_raw)
            answer = parse_answer_label(answer_raw)

            if options is None:
                reject_reason = "unparseable_options"
            elif answer is None:
                reject_reason = "unparseable_answer"
            else:
                prompt = (
                    f"{question}\n\n"
                    f"A) {options['A']}\n"
                    f"B) {options['B']}\n"
                    f"C) {options['C']}\n"
                    f"D) {options['D']}"
                )

                clean_rows.append(
                    {
                        "dataset": "cultsportqa",
                        "country": country,
                        "language": COUNTRY_TO_LANGUAGE[country],
                        "group": group,
                        "group_description": GROUP_DESCRIPTION.get(group, group),
                        "sport": sport,
                        "row_idx": int(row_idx) if pd.notna(row_idx) else None,
                        "source_file": source_file,
                        "source_url": source_url,
                        "question": question,
                        "option_A": options["A"],
                        "option_B": options["B"],
                        "option_C": options["C"],
                        "option_D": options["D"],
                        "prompt": prompt,
                        "answer": answer,
                        "raw_option": option_raw,
                        "raw_answer": answer_raw,
                    }
                )

        if reject_reason is not None:
            rejected_rows.append(
                {
                    "country": country,
                    "group": group,
                    "sport": sport,
                    "row_idx": int(row_idx) if pd.notna(row_idx) else None,
                    "question": question,
                    "option": option_raw,
                    "answer": answer_raw,
                    "reject_reason": reject_reason,
                }
            )

    clean_df = pd.DataFrame(clean_rows)

    if clean_df.empty:
        return clean_df

    clean_df["dedup_key"] = (
        clean_df["country"].map(normalize_for_dedup)
        + "\n"
        + clean_df["language"].map(normalize_for_dedup)
        + "\n"
        + clean_df["prompt"].map(normalize_for_dedup)
        + "\n"
        + clean_df["answer"].map(normalize_for_dedup)
    )

    clean_df = clean_df.drop_duplicates("dedup_key").copy()
    clean_df = clean_df.drop(columns=["dedup_key"])

    return clean_df


def make_cultsportqa_record(row, source_idx):
    answer = normalize_text(row["answer"]).upper()
    source_id = (
        f"cultsportqa_{row['language']}_"
        f"{int(row['row_idx']):06d}_{int(source_idx):06d}"
    )

    return {
        "id": source_id,
        "dataset": "cultsportqa",
        "language": normalize_text(row["language"]),
        "split": "train",
        "task_type": "mcq",
        "prompt": normalize_text(row["prompt"]),
        "answer": answer,
        "metadata": {
            "country": normalize_text(row.get("country", "")),
            "group": normalize_text(row.get("group", "")),
            "group_description": normalize_text(row.get("group_description", "")),
            "sport": normalize_text(row.get("sport", "")),
            "row_idx": int(row["row_idx"]) if pd.notna(row["row_idx"]) else None,
            "raw_option": normalize_text(row.get("raw_option", "")),
            "raw_answer": normalize_text(row.get("raw_answer", "")),
            "source_idx": int(source_idx),
            "source_file": normalize_text(row.get("source_file", "")),
            "source_url": normalize_text(row.get("source_url", "")),
        },
        "source_id": source_id,
    }


def validate_record(record):
    assert record["answer"] in VALID_ANSWERS
    assert record["prompt"]
    assert "\n\nA) " in record["prompt"]
    assert "\nB) " in record["prompt"]
    assert "\nC) " in record["prompt"]
    assert "\nD) " in record["prompt"]
    assert "\\boxed" not in record["prompt"]


def write_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw_dataframe()
    clean_df = build_clean_dataframe(raw_df)

    jsonl_records = []

    for source_idx, row in clean_df.iterrows():
        record = make_cultsportqa_record(row, source_idx)
        validate_record(record)
        jsonl_records.append(record)

    for language in sorted({record["language"] for record in jsonl_records}):
        lang_records = [
            record for record in jsonl_records if record["language"] == language
        ]
        out_path = OUTPUT_DIR / f"cultsportqa_{language}_all.jsonl"
        write_jsonl(lang_records, out_path)

    print(f"CultSportQA: wrote {len(jsonl_records)} records to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
