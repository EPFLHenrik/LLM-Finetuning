"""Preprocess Polizia di Stato into clean Italian MCQA JSONL."""

from pathlib import Path
from collections import Counter, defaultdict
import json
import re

import pandas as pd
from multilinguality_model.paths import get_processed_data_dir, get_data_dir


RAW_PATH = REPO_ROOT / "multilingual" / "data" / "raw" / "polizia_di_stato" / "polizia_di_stato_parsed.jsonl"
OUTPUT_DIR = get_processed_data_dir() / "polizia_di_stato"
OUTPUT_PATH = OUTPUT_DIR / "polizia_di_stato_initial_parsed.jsonl"

DATASET = "polizia_di_stato"
LANGUAGE = "italian"
SPLIT = "all"
VALID_ANSWERS = set("ABCDE")
KEEP_SECTIONS = {None, "", "CULTURA GENERALE"}

OPTION_LINE_RE = re.compile(r"(?m)^\s*([A-E])[\).]\s*(.*?)(?=^\s*[A-E][\).]\s*|\Z)", re.DOTALL)
BAD_EXACT_OPTION_VALUES = {"", ".", "-", "—", "–", "A", "I", "T"}


def read_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def clean_text(x):
    if x is None or pd.isna(x):
        return ""
    text = str(x).replace("\u00a0", " ")
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def normalize_for_dedup(text):
    text = clean_text(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\sàèéìòóù]", "", text, flags=re.UNICODE)
    return text.strip()


def get_section(record):
    metadata = record.get("metadata", {})
    section = metadata.get("section", record.get("section", None))
    if section is None:
        return None
    return clean_text(section).upper()


def section_is_kept(section):
    if section is None:
        return True
    return clean_text(section).upper() in KEEP_SECTIONS


def remove_isolated_watermark_lines(text):
    text = clean_text(text)
    lines = []
    for line in text.splitlines():
        if line.strip() in {"A", "I", "T"}:
            continue
        lines.append(line)
    return clean_text("\n".join(lines))


def clean_option_text(text):
    text = remove_isolated_watermark_lines(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_question_text(text):
    text = remove_isolated_watermark_lines(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_prompt(prompt):
    prompt = clean_text(prompt)
    matches = list(OPTION_LINE_RE.finditer(prompt))
    if not matches:
        return clean_question_text(prompt), {}
    question = prompt[:matches[0].start()].strip()
    options = {}
    for match in matches:
        label = match.group(1).strip().upper()
        value = clean_option_text(match.group(2))
        options[label] = value
    return clean_question_text(question), options


def extract_question_options(record):
    if "prompt" in record:
        question, options = parse_prompt(record["prompt"])
    else:
        question, options = "", {}
    if not question:
        for key in ["question", "domanda", "stem"]:
            if key in record:
                question = clean_question_text(record[key])
                break
    option_field_candidates = {
        "A": ["A", "option_a", "answer_a", "a"],
        "B": ["B", "option_b", "answer_b", "b"],
        "C": ["C", "option_c", "answer_c", "c"],
        "D": ["D", "option_d", "answer_d", "d"],
        "E": ["E", "option_e", "answer_e", "e"],
    }
    for label, keys in option_field_candidates.items():
        if label in options and options[label]:
            continue
        for key in keys:
            if key in record:
                value = clean_option_text(record[key])
                if value:
                    options[label] = value
                break
    if "options" in record:
        raw_options = record["options"]
        if isinstance(raw_options, dict):
            for label, value in raw_options.items():
                label = str(label).strip().upper()
                if label in VALID_ANSWERS and label not in options:
                    options[label] = clean_option_text(value)
        elif isinstance(raw_options, list):
            for idx, value in enumerate(raw_options):
                label = chr(ord("A") + idx)
                if label in VALID_ANSWERS and label not in options:
                    options[label] = clean_option_text(value)
    return question, dict(sorted(options.items()))


def get_answer(record):
    answer = clean_text(record.get("answer", record.get("correct_answer", record.get("label", "")))).upper()
    return answer.replace(")", "").replace(".", "").strip()


def make_prompt(question, options):
    labels = sorted(options.keys())
    return f"{question}\n\n" + "\n".join(f"{label}) {options[label]}" for label in labels)


def row_is_malformed(question, options, answer):
    if not question or len(question) < 8:
        return True, "bad_question"
    if answer not in VALID_ANSWERS:
        return True, "invalid_answer"
    if len(options) < 3:
        return True, "too_few_options"
    if answer not in options:
        return True, "answer_option_missing"
    expected_labels = [chr(ord("A") + i) for i in range(len(options))]
    actual_labels = sorted(options.keys())
    if actual_labels != expected_labels:
        return True, "non_contiguous_option_labels"
    for label, value in options.items():
        if label not in VALID_ANSWERS:
            return True, "invalid_option_label"
        if clean_text(value) in BAD_EXACT_OPTION_VALUES:
            return True, "bad_option_text"
        if len(clean_text(value)) < 2:
            return True, "option_too_short"
    option_values = [normalize_for_dedup(value) for value in options.values()]
    if len(option_values) != len(set(option_values)):
        return True, "duplicate_options"
    full_text = question + " " + " ".join(options.values())
    if full_text.count("�") >= 2:
        return True, "replacement_characters"
    return False, None


def build_clean_dataframe(raw_records):
    stats = Counter()
    clean_rows = []
    for source_idx, record in enumerate(raw_records):
        section = get_section(record)
        if not section_is_kept(section):
            stats[f"removed_section:{section}"] += 1
            continue
        question, options = extract_question_options(record)
        answer = get_answer(record)
        bad, reason = row_is_malformed(question, options, answer)
        if bad:
            stats[f"removed_{reason}"] += 1
            continue
        metadata = dict(record.get("metadata", {}))
        metadata.update({
            "source": "polizia_di_stato",
            "section": section,
            "original_id": record.get("id", record.get("original_id", source_idx)),
            "source_idx": source_idx,
            "n_options": len(options),
        })
        clean_rows.append({
            "source_idx": source_idx,
            "section": section,
            "question": question,
            "options": options,
            "prompt": make_prompt(question, options),
            "answer": answer,
            "metadata": metadata,
        })
    return pd.DataFrame(clean_rows), stats


def deduplicate_dataframe(df_in):
    df_out = df_in.copy()
    df_out["dedup_key"] = df_out["prompt"].apply(normalize_for_dedup)
    labels_by_key = defaultdict(set)
    for _, row in df_out.iterrows():
        labels_by_key[row["dedup_key"]].add(row["answer"])
    conflicting_keys = {key for key, labels in labels_by_key.items() if len(labels) > 1}
    df_out = df_out[~df_out["dedup_key"].isin(conflicting_keys)].copy()
    df_out = df_out.drop_duplicates(subset=["dedup_key", "answer"]).copy()
    return df_out.reset_index(drop=True)


def validate_record(record):
    assert record["dataset"] == DATASET
    assert record["language"] == LANGUAGE
    assert record["split"] == SPLIT
    assert record["task_type"] == "mcq"
    assert record["answer"] in VALID_ANSWERS
    assert record["prompt"].strip()
    assert "\n\nA) " in record["prompt"]
    assert "\nB) " in record["prompt"]
    assert record["answer"] + ")" in record["prompt"]
    assert isinstance(record["metadata"]["source_idx"], int)


def main():
    raw_records = read_jsonl(RAW_PATH)
    df, _ = build_clean_dataframe(raw_records)
    df = deduplicate_dataframe(df)
    records = []
    for idx, row in df.iterrows():
        records.append({
            "id": f"polizia_di_stato_italian_{idx}",
            "dataset": DATASET,
            "language": LANGUAGE,
            "split": SPLIT,
            "task_type": "mcq",
            "prompt": row["prompt"],
            "answer": row["answer"],
            "metadata": row["metadata"],
            "source_id": f"polizia_di_stato_{row['source_idx']}",
        })
    ids = [record["id"] for record in records]
    assert len(ids) == len(set(ids)), "Duplicate IDs found."
    prompt_keys = set()
    for record in records:
        validate_record(record)
        key = normalize_for_dedup(record["prompt"])
        assert key not in prompt_keys, "Duplicate prompt found."
        prompt_keys.add(key)
    write_jsonl(records, OUTPUT_PATH)
    print(f"Polizia di Stato: wrote {len(records)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
