"""Preprocess MMLU-ProX into clean multilingual MCQA JSONL files."""

from pathlib import Path
from collections import defaultdict
import hashlib
import json
import random

from datasets import get_dataset_config_names, load_dataset
from multilinguality_model.paths import get_processed_data_dir, get_data_dir


OUT_DIR = REPO_ROOT / "multilingual" / "data" / "processed" / "mmlu_prox"

HF_DATASET = "li-lab/MMLU-ProX"
DATASET_NAME = "mmlu_prox"

LANG_CONFIGS = {
    "russian": "ru",
    "chinese": "zh",
    "hindi": "hi",
    "italian": "it",
    "spanish": "es",
}

LETTERS = list("ABCDEFGHIJ")
EVAL_FRACTION = 0.15
SEED = 552
SUBSET_N_PER_LANGUAGE = 5000

SUBSET_SEEDS = {
    "russian": 5521,
    "chinese": 5522,
    "hindi": 5523,
    "italian": 5524,
    "spanish": 5525,
}

SUBJECT_WEIGHTS = {
    "law": 1.6,
    "history": 1.6,
    "economics": 1.6,
    "business": 1.6,
    "philosophy": 1.6,
    "psychology": 1.6,
    "other": 1.6,
    "math": 1.0,
    "physics": 1.0,
    "chemistry": 1.0,
    "engineering": 1.0,
    "computer science": 1.0,
    "biology": 1.4,
    "health": 1.4,
}


def write_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_options(row):
    options = []
    for i in range(10):
        key = f"option_{i}"
        value = row.get(key, None)
        if value is None:
            continue
        value = str(value).strip()
        if not value or value.lower() in {"none", "null"}:
            continue
        options.append({"original_index": i, "text": value})
    if len(options) < 3:
        return None
    return options


def normalize_answer(row, options):
    answer_index = row.get("answer_index", None)
    if isinstance(answer_index, int):
        idx = answer_index
    elif str(answer_index).isdigit():
        idx = int(answer_index)
    else:
        return None
    original_indices = [opt["original_index"] for opt in options]
    if idx not in original_indices:
        return None
    new_index = original_indices.index(idx)
    if new_index >= len(LETTERS):
        return None
    return LETTERS[new_index]


def format_prompt(question, options):
    options_text = "\n".join(f"{LETTERS[i]}) {opt['text']}" for i, opt in enumerate(options))
    return f"{question.strip()}\n\n{options_text}"


def make_source_id(language, row, hf_split):
    if row.get("question_id_src", None) not in (None, ""):
        return f"mmlu_prox_{language}_{hf_split}_{row['question_id_src']}"
    if row.get("question_id", None) not in (None, ""):
        return f"mmlu_prox_{language}_{hf_split}_{row['question_id']}"
    raw = f"{language}\n{hf_split}\n{row.get('question', '')}\n{row.get('answer', '')}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"mmlu_prox_{language}_{hf_split}_{digest}"


def normalize_record(row, language, hf_split):
    row = dict(row)
    question = str(row.get("question", "")).strip()
    if not question:
        return None
    options = normalize_options(row)
    if options is None:
        return None
    answer = normalize_answer(row, options)
    if answer not in LETTERS[:len(options)]:
        return None
    source_id = make_source_id(language, row, hf_split)
    subject = str(row.get("category", "unknown")).strip() or "unknown"
    return {
        "source_id": source_id,
        "language": language,
        "dataset": DATASET_NAME,
        "prompt": format_prompt(question, options),
        "answer": answer,
        "id": source_id,
        "split": "all",
        "task_type": "mcq",
        "metadata": {
            "source_dataset": HF_DATASET,
            "hf_config": LANG_CONFIGS[language],
            "hf_split": hf_split,
            "question_id": row.get("question_id"),
            "question_id_src": row.get("question_id_src"),
            "subject": subject,
            "category": subject,
            "src": row.get("src"),
            "answer_index": row.get("answer_index"),
            "n_options": len(options),
            "option_letters": LETTERS[:len(options)],
            "original_option_indices": [opt["original_index"] for opt in options],
        },
    }


def dedupe_by_source_id(records):
    seen = set()
    out = []
    for record in records:
        source_id = record["source_id"]
        if source_id in seen:
            continue
        seen.add(source_id)
        out.append(record)
    return out


def load_records_by_language():
    configs = get_dataset_config_names(HF_DATASET)
    missing = [cfg for cfg in LANG_CONFIGS.values() if cfg not in configs]
    assert not missing, f"Missing MMLU-ProX configs: {missing}"
    records_by_language = {}
    for language, cfg in LANG_CONFIGS.items():
        ds = load_dataset(HF_DATASET, cfg)
        records = []
        for hf_split in ds:
            for row in ds[hf_split]:
                record = normalize_record(row, language, hf_split)
                if record is not None:
                    records.append(record)
        records = dedupe_by_source_id(records)
        assert records
        records_by_language[language] = records
    return records_by_language


def get_base_id(record):
    qid_src = record["metadata"].get("question_id_src")
    if qid_src not in (None, ""):
        return f"question_id_src_{qid_src}"
    qid = record["metadata"].get("question_id")
    if qid not in (None, ""):
        return f"question_id_{qid}"
    return record["source_id"]


def grouped_train_eval_split(records_by_language, eval_fraction=0.15, seed=552):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for language, records in records_by_language.items():
        for record in records:
            base_id = get_base_id(record)
            subject = record["metadata"].get("subject", "unknown")
            n_options = record["metadata"].get("n_options", 10)
            groups[(base_id, subject, n_options)].append(record)
    strata = defaultdict(list)
    for (base_id, subject, n_options), group_records in groups.items():
        strata[(subject, n_options)].append((base_id, group_records))
    train_by_language = {language: [] for language in records_by_language}
    eval_by_language = {language: [] for language in records_by_language}
    for group_list in strata.values():
        group_list = list(group_list)
        rng.shuffle(group_list)
        if len(group_list) >= 10:
            n_eval = max(1, int(round(len(group_list) * eval_fraction)))
        elif len(group_list) >= 2:
            n_eval = 1
        else:
            n_eval = 0
        eval_groups = group_list[:n_eval]
        train_groups = group_list[n_eval:]
        for _, group_records in train_groups:
            for record in group_records:
                out = dict(record)
                out["metadata"] = dict(out["metadata"])
                out["split"] = "train"
                out["metadata"]["local_split"] = "train"
                out["metadata"]["base_question_id"] = get_base_id(record)
                train_by_language[out["language"]].append(out)
        for _, group_records in eval_groups:
            for record in group_records:
                out = dict(record)
                out["metadata"] = dict(out["metadata"])
                out["split"] = "eval"
                out["metadata"]["local_split"] = "eval"
                out["metadata"]["base_question_id"] = get_base_id(record)
                eval_by_language[out["language"]].append(out)
    for language in records_by_language:
        rng.shuffle(train_by_language[language])
        rng.shuffle(eval_by_language[language])
    return train_by_language, eval_by_language


def weighted_sample_without_replacement(records, n, seed):
    rng = random.Random(seed)
    remaining = list(records)
    selected = []
    while remaining and len(selected) < n:
        weights = [SUBJECT_WEIGHTS.get(record["metadata"].get("subject", "unknown"), 1.0) for record in remaining]
        chosen = rng.choices(remaining, weights=weights, k=1)[0]
        selected.append(chosen)
        remaining.remove(chosen)
    rng.shuffle(selected)
    return selected


def save_main_outputs(records_by_language, train_by_language, eval_by_language):
    for language in LANG_CONFIGS:
        all_records_out = []
        for record in records_by_language[language]:
            out = dict(record)
            out["split"] = "all"
            out["metadata"] = dict(out["metadata"])
            out["metadata"].pop("local_split", None)
            if "base_question_id" not in out["metadata"]:
                out["metadata"]["base_question_id"] = get_base_id(out)
            all_records_out.append(out)
        write_jsonl(all_records_out, OUT_DIR / f"mmlu_prox_{language}_all.jsonl")
        write_jsonl(train_by_language[language], OUT_DIR / f"mmlu_prox_{language}_train.jsonl")
        write_jsonl(eval_by_language[language], OUT_DIR / f"mmlu_prox_{language}_eval.jsonl")
    all_train = []
    all_eval = []
    for language in LANG_CONFIGS:
        all_train.extend(train_by_language[language])
        all_eval.extend(eval_by_language[language])
    random.Random(SEED).shuffle(all_train)
    random.Random(SEED).shuffle(all_eval)
    write_jsonl(all_train, OUT_DIR / "mmlu_prox_all_train.jsonl")
    write_jsonl(all_eval, OUT_DIR / "mmlu_prox_all_eval.jsonl")


def save_5k_subsets(train_by_language, eval_by_language):
    subset_dir = OUT_DIR / "subsets"
    subset_dir.mkdir(parents=True, exist_ok=True)
    train_5k_by_language = {}
    for language in LANG_CONFIGS:
        subset = weighted_sample_without_replacement(
            list(train_by_language[language]),
            n=min(SUBSET_N_PER_LANGUAGE, len(train_by_language[language])),
            seed=SUBSET_SEEDS[language],
        )
        train_5k_by_language[language] = subset
        write_jsonl(subset, subset_dir / f"mmlu_prox_{language}_train_5k.jsonl")
    combined = []
    for language in LANG_CONFIGS:
        combined.extend(train_5k_by_language[language])
    random.Random(5520).shuffle(combined)
    write_jsonl(combined, subset_dir / "mmlu_prox_all_train_5k_per_language.jsonl")
    eval_base_ids = set()
    for language in LANG_CONFIGS:
        for record in eval_by_language[language]:
            eval_base_ids.add(record["metadata"]["base_question_id"])
    for language in LANG_CONFIGS:
        subset_base_ids = {record["metadata"]["base_question_id"] for record in train_5k_by_language[language]}
        assert not (subset_base_ids & eval_base_ids)
    assert not ({record["metadata"]["base_question_id"] for record in combined} & eval_base_ids)


def validate_splits(train_by_language, eval_by_language):
    train_base_ids = set()
    eval_base_ids = set()
    for language in LANG_CONFIGS:
        for record in train_by_language[language]:
            train_base_ids.add(record["metadata"]["base_question_id"])
        for record in eval_by_language[language]:
            eval_base_ids.add(record["metadata"]["base_question_id"])
    assert not (train_base_ids & eval_base_ids)
    for language in LANG_CONFIGS:
        train_ids = {record["source_id"] for record in train_by_language[language]}
        eval_ids = {record["source_id"] for record in eval_by_language[language]}
        assert not (train_ids & eval_ids)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records_by_language = load_records_by_language()
    train_by_language, eval_by_language = grouped_train_eval_split(records_by_language, eval_fraction=EVAL_FRACTION, seed=SEED)
    validate_splits(train_by_language, eval_by_language)
    save_main_outputs(records_by_language, train_by_language, eval_by_language)
    save_5k_subsets(train_by_language, eval_by_language)
    print(f"MMLU-ProX: wrote outputs for {len(LANG_CONFIGS)} languages to {OUT_DIR}")


if __name__ == "__main__":
    main()
