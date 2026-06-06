"""Preprocess HEAD-QA into clean multilingual MCQA JSONL files."""

from pathlib import Path
import json
import random
import re

import pandas as pd
from datasets import load_dataset


REPO_ROOT = Path("/scratch/workspace/emma/standard-project-m2-mystery-machine")
OUTPUT_DIR = REPO_ROOT / "multilingual" / "data" / "processed" / "head_qa"

TARGET_CONFIGS = {"es": "spanish", "it": "italian"}
LETTERS = "ABCDE"
LETTERS_LONG = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
RANDOM_SEED = 42
N_EVAL_SPANISH = 200
CORRUPTION_RE = re.compile(r"[\ufffd\x00-\x08\x0b\x0c\x0e-\x1f]")


def has_corruption(row):
    return any(CORRUPTION_RE.search(str(field)) for field in [row["qtext"]] + list(row["atexts"]))


def make_headqa_prompt(row):
    options = row["atexts"]
    if isinstance(options, str):
        options = json.loads(options)
    lines = [str(row["qtext"]).strip(), ""]
    for i, option in enumerate(options):
        lines.append(f"{LETTERS_LONG[i]}) {str(option).strip()}")
    return "\n".join(lines)


def safe_id_part(x):
    return str(x).strip().replace(" ", "_").replace("/", "_").replace("\\", "_")


def make_headqa_record(row, source_idx):
    options = row["atexts"]
    if isinstance(options, str):
        options = json.loads(options)
    answer = str(row["ra_letter"]).strip().upper()
    config = safe_id_part(row["config"])
    language = safe_id_part(row["language"])
    name = safe_id_part(row["name"])
    qid = int(row["qid"])
    source_id = f"head_qa_{language}_{config}_{name}_{qid:06d}_{int(source_idx):06d}"
    return {
        "id": source_id,
        "dataset": "head_qa",
        "language": row["language"],
        "split": "train",
        "task_type": "mcq",
        "prompt": make_headqa_prompt(row),
        "answer": answer,
        "metadata": {
            "config": row["config"],
            "qid": qid,
            "name": row["name"],
            "year": int(row["year"]),
            "category": row["category"],
            "n_answers": len(options),
            "ra": int(row["ra"]),
            "ra_letter": answer,
            "has_image": bool(row["has_image"]),
            "aids": row["aids"] if not isinstance(row["aids"], str) else json.loads(row["aids"]),
            "source_idx": int(source_idx),
        },
        "source_id": source_id,
    }


def write_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_raw_dataframe():
    datasets = {config: load_dataset("alesi12/head_qa_v2", config) for config in TARGET_CONFIGS}
    rows = []
    for config, language in TARGET_CONFIGS.items():
        for ex in datasets[config]["train"]:
            answers = ex["answers"]
            ra = int(ex["ra"])
            has_image = ex["image"] is not None
            ra_idx = next((i for i, answer in enumerate(answers) if answer["aid"] == ra), None)
            ra_letter = LETTERS[ra_idx] if ra_idx is not None else None
            rows.append({
                "config": config,
                "language": language,
                "qid": ex["qid"],
                "name": ex["name"],
                "year": ex["year"],
                "category": ex["category"],
                "qtext": ex["qtext"],
                "n_answers": len(answers),
                "ra": ra,
                "ra_letter": ra_letter,
                "has_image": has_image,
                "aids": [answer["aid"] for answer in answers],
                "atexts": [answer["atext"] for answer in answers],
            })
    df = pd.DataFrame(rows)
    df["has_corruption"] = df.apply(has_corruption, axis=1)
    return df


def clean_dataframe(df):
    return df[
        ~df["has_image"]
        & df["ra_letter"].notna()
        & ~df["has_corruption"]
        & (df["qtext"].str.strip() != "")
    ].copy()


def build_jsonl_records(clean):
    records = []
    bad_rows = []
    for source_idx, row in clean.iterrows():
        try:
            options = row["atexts"]
            if isinstance(options, str):
                options = json.loads(options)
            record = make_headqa_record(row, source_idx)
            n_options = len(options)
            assert 2 <= n_options <= len(LETTERS_LONG)
            assert record["answer"] in LETTERS_LONG[:n_options]
            assert record["prompt"].count("\n") >= n_options
            assert "\\boxed" not in record["prompt"]
            records.append(record)
        except Exception as exc:
            bad_rows.append({
                "qid": row.get("qid"),
                "language": row.get("language"),
                "name": row.get("name"),
                "n_answers": row.get("n_answers"),
                "error": repr(exc),
            })
    assert not bad_rows, f"Bad HEAD-QA rows: {bad_rows[:5]}"
    return records


def split_records(jsonl_records):
    records_by_language = {
        lang_name: [record for record in jsonl_records if record["metadata"]["config"] == config]
        for config, lang_name in TARGET_CONFIGS.items()
    }
    spanish_records = records_by_language["spanish"]
    assert len(spanish_records) >= N_EVAL_SPANISH
    rng = random.Random(RANDOM_SEED)
    indices = list(range(len(spanish_records)))
    rng.shuffle(indices)
    eval_indices = set(indices[:N_EVAL_SPANISH])
    spanish_eval_records = []
    spanish_train_records = []
    for i, record in enumerate(spanish_records):
        record = dict(record)
        if i in eval_indices:
            record["split"] = "eval"
            spanish_eval_records.append(record)
        else:
            record["split"] = "train"
            spanish_train_records.append(record)
    assert len(spanish_eval_records) == N_EVAL_SPANISH
    assert len(spanish_eval_records) + len(spanish_train_records) == len(spanish_records)
    assert {record["id"] for record in spanish_eval_records}.isdisjoint({record["id"] for record in spanish_train_records})
    italian_train_records = []
    for record in records_by_language["italian"]:
        record = dict(record)
        record["split"] = "train"
        italian_train_records.append(record)
    return spanish_eval_records, spanish_train_records, italian_train_records


def validate_saved_records(records):
    seen = set()
    for record in records:
        assert record["dataset"] == "head_qa"
        assert record["language"] in {"spanish", "italian"}
        assert record["split"] in {"train", "eval"}
        assert record["task_type"] == "mcq"
        assert record["source_id"] not in seen
        assert "\n\nA) " in record["prompt"]
        assert f"{record['answer']}) " in record["prompt"]
        n_options = record["metadata"]["n_answers"]
        assert record["answer"] in LETTERS_LONG[:n_options]
        seen.add(record["source_id"])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_raw_dataframe()
    clean = clean_dataframe(df)
    jsonl_records = build_jsonl_records(clean)
    spanish_eval, spanish_train, italian_train = split_records(jsonl_records)
    outputs = {
        "head_qa_spanish_eval.jsonl": spanish_eval,
        "head_qa_spanish_train.jsonl": spanish_train,
        "head_qa_italian_train.jsonl": italian_train,
    }
    for filename, records in outputs.items():
        validate_saved_records(records)
        write_jsonl(records, OUTPUT_DIR / filename)
    print(f"HEAD-QA: wrote {sum(len(records) for records in outputs.values())} records to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
