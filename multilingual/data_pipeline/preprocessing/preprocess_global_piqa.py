"""Preprocess Global PIQA into clean multilingual MCQA JSONL files."""

from pathlib import Path
from collections import Counter, defaultdict
import json
import re

from datasets import get_dataset_config_names, load_dataset, concatenate_datasets


DATASET_NAME = "mrlbenchmarks/global-piqa-nonparallel"
REPO_ROOT = Path("/scratch/workspace/emma/standard-project-m2-mystery-machine")
OUTPUT_DIR = REPO_ROOT / "multilingual" / "data" / "processed" / "global_piqa"

SEED = 42
BAD_TEXT_VALUES = {"", "nan", "none", "null", "n/a", "na", "?", "-", "undefined"}
TIMESTAMP_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\s+00:00:00\b")
ARTIFACT_RE = re.compile(r"(Unnamed:\s*\d+|\[object Object\])", re.IGNORECASE)

GLOBAL_PIQA_CONFIGS = {
    "chinese": ["cmn_hans", "cmn_hant"],
    "hindi": ["hin_deva", "hin_latn"],
    "italian": ["ita_latn"],
    "spanish": ["spa_latn_mexi", "spa_latn_peru", "spa_latn_spai"],
    "russian": ["rus_cyrl"],
}


def normalize_text(x):
    return re.sub(r"\s+", " ", str(x).strip()).lower()


def text_is_missing_or_bad(x):
    if x is None:
        return True
    text = str(x).strip()
    low = text.lower()
    if low in BAD_TEXT_VALUES:
        return True
    if ARTIFACT_RE.search(text):
        return True
    if TIMESTAMP_RE.search(text):
        return True
    return False


def label_is_valid(label):
    try:
        return int(label) in {0, 1}
    except Exception:
        return False


def row_is_malformed(row):
    for key in ["prompt", "solution0", "solution1", "label"]:
        if key not in row:
            return True, f"missing_field:{key}"
    if text_is_missing_or_bad(row["prompt"]):
        return True, "bad_prompt"
    if text_is_missing_or_bad(row["solution0"]):
        return True, "bad_solution0"
    if text_is_missing_or_bad(row["solution1"]):
        return True, "bad_solution1"
    if not label_is_valid(row["label"]):
        return True, "invalid_label"
    if normalize_text(row["solution0"]) == normalize_text(row["solution1"]):
        return True, "duplicate_options"
    if len(str(row["prompt"]).strip()) < 5:
        return True, "prompt_too_short"
    if len(str(row["solution0"]).strip()) < 2:
        return True, "solution0_too_short"
    if len(str(row["solution1"]).strip()) < 2:
        return True, "solution1_too_short"
    return False, None


def row_key(row):
    return normalize_text(row["prompt"]), normalize_text(row["solution0"]), normalize_text(row["solution1"])


def clean_dataset_rows(rows):
    stats = Counter()
    valid_rows = []
    for row in rows:
        bad, reason = row_is_malformed(row)
        if bad:
            stats[f"removed_{reason}"] += 1
            continue
        valid_rows.append(row)
    seen_example_ids = set()
    dedup_by_id = []
    for row in valid_rows:
        ex_id = row.get("example_id")
        if ex_id is not None:
            ex_id = str(ex_id).strip()
            if ex_id in seen_example_ids:
                stats["removed_duplicate_example_id"] += 1
                continue
            seen_example_ids.add(ex_id)
        dedup_by_id.append(row)
    labels_by_key = defaultdict(set)
    for row in dedup_by_id:
        labels_by_key[row_key(row)].add(int(row["label"]))
    conflicting_keys = {key for key, labels in labels_by_key.items() if len(labels) > 1}
    no_conflicts = []
    for row in dedup_by_id:
        if row_key(row) in conflicting_keys:
            stats["removed_conflicting_duplicate"] += 1
            continue
        no_conflicts.append(row)
    seen_full = set()
    cleaned = []
    for row in no_conflicts:
        full_key = row_key(row) + (int(row["label"]),)
        if full_key in seen_full:
            stats["removed_duplicate_prompt_options_label"] += 1
            continue
        seen_full.add(full_key)
        cleaned.append(row)
    stats["kept"] = len(cleaned)
    stats["input"] = len(rows)
    stats["removed_total"] = len(rows) - len(cleaned)
    return cleaned, stats


def label_to_letter(label):
    if int(label) == 0:
        return "A"
    if int(label) == 1:
        return "B"
    raise ValueError(f"Unexpected label: {label}")


def make_prompt(row):
    return f"{row['prompt']}\n\nA) {row['solution0']}\nB) {row['solution1']}"


def convert_example(row, idx, language, config_name):
    metadata = {"config": config_name}
    for key in ["approx_cultural_score", "example_id", "source_split"]:
        if key in row and row[key] is not None:
            metadata[key] = row[key]
    return {
        "id": f"global_piqa_{language}_{idx}",
        "dataset": "global_piqa",
        "language": language,
        "task_type": "mcq",
        "prompt": make_prompt(row),
        "answer": label_to_letter(row["label"]),
        "metadata": metadata,
    }


def load_all_splits_for_config(config_name):
    ds_dict = load_dataset(DATASET_NAME, config_name)
    splits = []
    for split_name, split_ds in ds_dict.items():
        split_ds = split_ds.add_column("source_split", [split_name] * len(split_ds))
        splits.append(split_ds)
    return concatenate_datasets(splits)


def write_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_record(record):
    assert record["dataset"] == "global_piqa"
    assert record["language"] in GLOBAL_PIQA_CONFIGS
    assert record["task_type"] == "mcq"
    assert record["answer"] in {"A", "B"}
    assert "\n\nA) " in record["prompt"]
    assert "\nB) " in record["prompt"]
    assert isinstance(record["metadata"], dict)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    available_configs = set(get_dataset_config_names(DATASET_NAME))
    requested_configs = {cfg for configs in GLOBAL_PIQA_CONFIGS.values() for cfg in configs}
    missing_configs = sorted(requested_configs - available_configs)
    assert not missing_configs, f"Missing Global PIQA configs: {missing_configs}"
    for language, configs in GLOBAL_PIQA_CONFIGS.items():
        datasets_for_language = []
        for cfg in configs:
            ds = load_all_splits_for_config(cfg)
            ds = ds.add_column("config_name", [cfg] * len(ds))
            datasets_for_language.append(ds)
        combined = concatenate_datasets(datasets_for_language)
        combined = combined.shuffle(seed=SEED)
        rows = [dict(row) for row in combined]
        cleaned_rows, _ = clean_dataset_rows(rows)
        records = [
            convert_example(row=row, idx=idx, language=language, config_name=row["config_name"])
            for idx, row in enumerate(cleaned_rows)
        ]
        ids = [record["id"] for record in records]
        assert len(ids) == len(set(ids)), f"Duplicate ids for {language}"
        for record in records:
            validate_record(record)
        write_jsonl(records, OUTPUT_DIR / f"global_piqa_{language}_all.jsonl")
    print(f"Global PIQA: wrote outputs for {len(GLOBAL_PIQA_CONFIGS)} languages to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
