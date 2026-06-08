"""Preprocess EXAMS into clean multilingual MCQA JSONL files."""

from pathlib import Path
from collections import Counter
import json
import random
import re

import pandas as pd
from datasets import load_dataset
from multilinguality_model.paths import get_processed_data_dir, get_data_dir


DATASET_NAME = "mhardalov/exams"
CONFIG_NAME = "multilingual"
OUT_DIR = REPO_ROOT / "multilingual" / "data" / "processed" / "exams"

TARGET_LANGUAGES = {
    "Italian": "italian",
    "Spanish": "spanish",
}
BAD_TEXT_VALUES = {"", "nan", "none", "null", "n/a", "na", "?", "-", "undefined"}
CORRUPTION_RE = re.compile(r"[\ufffd\x00-\x08\x0b\x0c\x0e-\x1f]")
ARTIFACT_RE = re.compile(r"(Unnamed:\s*\d+|\[object Object\])", re.IGNORECASE)
TIMESTAMP_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\s+00:00:00\b")
RANDOM_SEED = 42
N_EVAL_ITALIAN = 200


def normalize_text(x):
    return re.sub(r"\s+", " ", str(x).strip()).lower()


def is_bad_text(x):
    if x is None:
        return True
    text = str(x).strip()
    low = text.lower()
    if low in BAD_TEXT_VALUES:
        return True
    if CORRUPTION_RE.search(text):
        return True
    if ARTIFACT_RE.search(text):
        return True
    if TIMESTAMP_RE.search(text):
        return True
    return False


def has_nonempty_para(paras):
    if not isinstance(paras, list):
        return False
    return any(str(p).strip() for p in paras if p is not None)


def row_key(row):
    return normalize_text(row["stem"]), tuple(normalize_text(x) for x in row["texts"])


def full_row_key(row):
    return row_key(row) + (str(row["answerKey"]).strip(),)


def validate_row(row):
    labels = row["labels"]
    texts = row["texts"]
    answer = row["answerKey"]
    if row["language"] not in TARGET_LANGUAGES.values():
        return False, "not_target_language"
    if row["source_split"] not in {"train", "validation", "test"}:
        return False, "bad_split"
    if is_bad_text(row["stem"]):
        return False, "bad_stem"
    if not isinstance(labels, list) or not isinstance(texts, list):
        return False, "bad_choices_type"
    if len(labels) != len(texts):
        return False, "label_text_length_mismatch"
    if len(labels) < 2:
        return False, "too_few_choices"
    if answer not in labels:
        return False, "answer_not_in_labels"
    if any(is_bad_text(t) for t in texts):
        return False, "bad_choice_text"
    if len(set(str(label).strip() for label in labels)) != len(labels):
        return False, "duplicate_labels"
    if len(set(normalize_text(text) for text in texts)) != len(texts):
        return False, "duplicate_choice_texts"
    return True, None


def format_prompt(stem, labels, texts):
    option_lines = [f"{str(label).strip()}) {str(text).strip()}" for label, text in zip(labels, texts)]
    return str(stem).strip() + "\n\n" + "\n".join(option_lines)


def format_jsonl_example(row, idx):
    language = row["language"]
    labels = row["labels"]
    texts = row["texts"]
    answer = str(row["answerKey"]).strip()
    return {
        "id": f"exams_{language}_{idx}",
        "dataset": "exams",
        "language": language,
        "task_type": "mcq",
        "prompt": format_prompt(row["stem"], labels, texts),
        "answer": answer,
        "metadata": {
            "source_id": row.get("source_id", ""),
            "source_split": row.get("source_split", ""),
            "language_raw": row.get("language_raw", ""),
            "subject": row.get("subject", ""),
            "grade": row.get("grade", ""),
            "n_choices": len(labels),
            "has_para": has_nonempty_para(row.get("paras", [])),
        },
    }


def write_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_raw_dataframe():
    ds = load_dataset(DATASET_NAME, CONFIG_NAME)
    rows = []
    for split_name in ds.keys():
        for ex in ds[split_name]:
            info = ex.get("info", {}) or {}
            question = ex.get("question", {}) or {}
            choices = question.get("choices", {}) or {}
            labels = choices.get("label", []) or []
            texts = choices.get("text", []) or []
            paras = choices.get("para", []) or []
            rows.append({
                "source_split": split_name,
                "source_id": ex.get("id"),
                "language_raw": info.get("language"),
                "subject": info.get("subject"),
                "grade": info.get("grade"),
                "stem": question.get("stem"),
                "labels": labels,
                "texts": texts,
                "paras": paras,
                "answerKey": ex.get("answerKey"),
                "n_choices": len(labels),
            })
    return pd.DataFrame(rows)


def clean_dataframe(df):
    df_target = df[df["language_raw"].isin(TARGET_LANGUAGES.keys())].copy()
    df_target["language"] = df_target["language_raw"].map(TARGET_LANGUAGES)
    valid_rows = []
    rejected_rows = []
    stats = Counter()
    for _, row in df_target.iterrows():
        ok, reason = validate_row(row)
        row_dict = row.to_dict()
        if ok:
            valid_rows.append(row_dict)
        else:
            row_dict["rejection_reason"] = reason
            rejected_rows.append(row_dict)
            stats[f"removed_{reason}"] += 1
    answers_by_key = {}
    for row in valid_rows:
        key = row_key(row)
        answers_by_key.setdefault(key, set()).add(str(row["answerKey"]).strip())
    conflicting_keys = {key for key, answers in answers_by_key.items() if len(answers) > 1}
    nonconflicting_rows = []
    for row in valid_rows:
        if row_key(row) in conflicting_keys:
            row["rejection_reason"] = "conflicting_duplicate"
            rejected_rows.append(row)
            stats["removed_conflicting_duplicate"] += 1
        else:
            nonconflicting_rows.append(row)
    seen = set()
    clean_rows = []
    for row in nonconflicting_rows:
        key = full_row_key(row)
        if key in seen:
            row["rejection_reason"] = "exact_duplicate"
            rejected_rows.append(row)
            stats["removed_exact_duplicate"] += 1
        else:
            seen.add(key)
            clean_rows.append(row)
    return pd.DataFrame(clean_rows), pd.DataFrame(rejected_rows), stats


def validate_record(record):
    assert record["dataset"] == "exams"
    assert record["language"] in {"italian", "spanish"}
    assert record["split"] in {"train", "eval"}
    assert record["task_type"] == "mcq"
    assert isinstance(record["prompt"], str)
    assert record["prompt"].strip()
    assert record["answer"] in record["prompt"], record["id"]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_raw_dataframe()
    clean, _, _ = clean_dataframe(df)
    jsonl_examples_by_language = {}
    for language in TARGET_LANGUAGES.values():
        lang_df = clean[clean["language"] == language].copy()
        jsonl_examples_by_language[language] = [format_jsonl_example(row.to_dict(), idx) for idx, (_, row) in enumerate(lang_df.iterrows())]
    italian_records = jsonl_examples_by_language["italian"]
    assert len(italian_records) >= N_EVAL_ITALIAN
    rng = random.Random(RANDOM_SEED)
    indices = list(range(len(italian_records)))
    rng.shuffle(indices)
    eval_indices = set(indices[:N_EVAL_ITALIAN])
    italian_eval_records = []
    italian_train_records = []
    for i, record in enumerate(italian_records):
        record = dict(record)
        if i in eval_indices:
            record["split"] = "eval"
            italian_eval_records.append(record)
        else:
            record["split"] = "train"
            italian_train_records.append(record)
    assert len(italian_eval_records) == N_EVAL_ITALIAN
    assert len(italian_eval_records) + len(italian_train_records) == len(italian_records)
    assert {record["id"] for record in italian_eval_records}.isdisjoint({record["id"] for record in italian_train_records})
    spanish_train_records = []
    for record in jsonl_examples_by_language["spanish"]:
        record = dict(record)
        record["split"] = "train"
        spanish_train_records.append(record)
    paths = [
        OUT_DIR / "exams_italian_eval.jsonl",
        OUT_DIR / "exams_italian_train.jsonl",
        OUT_DIR / "exams_spanish_train.jsonl",
    ]
    write_jsonl(italian_eval_records, paths[0])
    write_jsonl(italian_train_records, paths[1])
    write_jsonl(spanish_train_records, paths[2])
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            loaded = [json.loads(line) for line in f if line.strip()]
        ids = [record["id"] for record in loaded]
        assert len(ids) == len(set(ids)), f"Duplicate ids in {path}"
        for record in loaded:
            validate_record(record)
    print(f"EXAMS: wrote {len(italian_eval_records) + len(italian_train_records) + len(spanish_train_records)} records to {OUT_DIR}")


if __name__ == "__main__":
    main()
