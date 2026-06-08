"""code/multilinguality_model/data_pipeline/preprocessing/build_benchmark.py

Build multilingual benchmark files from Hugging Face processed data pools.
"""

from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import random
import shutil

from huggingface_hub import snapshot_download

from multilinguality_model.paths import get_processed_data_dir, get_benchmarks_data_dir


HF_DATASET_REPO = "cs-552-2026-mystery-machine/multilingual"
HF_PROCESSED_SUBDIR = "processed_data"

PROCESSED_DIR = get_processed_data_dir()
BENCHMARK_DIR = get_benchmarks_data_dir()
BY_LANGUAGE_DIR = BENCHMARK_DIR / "by_language"
BY_DATASET_DIR = BENCHMARK_DIR / "by_dataset"
ALL_PATH = BENCHMARK_DIR / "benchmark_all.jsonl"
MANIFEST_PATH = BENCHMARK_DIR / "manifest.json"

RANDOM_SEED = 42
VALID_ANSWER_LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

CANDIDATE_FILES = {
    ("include", "russian"): [PROCESSED_DIR / "include" / "include_russian_all.jsonl"],
    ("include", "chinese"): [PROCESSED_DIR / "include" / "include_chinese_all.jsonl"],
    ("include", "italian"): [PROCESSED_DIR / "include" / "include_italian_all.jsonl"],
    ("include", "spanish"): [PROCESSED_DIR / "include" / "include_spanish_all.jsonl"],
    ("include", "hindi"): [PROCESSED_DIR / "include" / "include_hindi_all.jsonl"],

    ("global_mmlu", "russian"): [PROCESSED_DIR / "global_mmlu" / "global_mmlu_russian_all.jsonl"],
    ("global_mmlu", "chinese"): [PROCESSED_DIR / "global_mmlu" / "global_mmlu_chinese_all.jsonl"],
    ("global_mmlu", "italian"): [PROCESSED_DIR / "global_mmlu" / "global_mmlu_italian_all.jsonl"],
    ("global_mmlu", "spanish"): [PROCESSED_DIR / "global_mmlu" / "global_mmlu_spanish_all.jsonl"],
    ("global_mmlu", "hindi"): [PROCESSED_DIR / "global_mmlu" / "global_mmlu_hindi_all.jsonl"],

    ("global_piqa", "russian"): [PROCESSED_DIR / "global_piqa" / "global_piqa_russian_all.jsonl"],
    ("global_piqa", "chinese"): [PROCESSED_DIR / "global_piqa" / "global_piqa_chinese_all.jsonl"],
    ("global_piqa", "italian"): [PROCESSED_DIR / "global_piqa" / "global_piqa_italian_all.jsonl"],
    ("global_piqa", "spanish"): [PROCESSED_DIR / "global_piqa" / "global_piqa_spanish_all.jsonl"],
    ("global_piqa", "hindi"): [PROCESSED_DIR / "global_piqa" / "global_piqa_hindi_all.jsonl"],

    ("ceval", "chinese"): [PROCESSED_DIR / "ceval" / "ceval_chinese_all.jsonl"],
    ("cmmlu", "chinese"): [PROCESSED_DIR / "cmmlu" / "cmmlu_chinese_all.jsonl"],
    ("cultsportqa", "chinese"): [PROCESSED_DIR / "cultsportqa" / "cultsportqa_chinese_all.jsonl"],
    ("cultsportqa", "hindi"): [PROCESSED_DIR / "cultsportqa" / "cultsportqa_hindi_all.jsonl"],

    ("milu", "hindi"): [PROCESSED_DIR / "milu" / "milu_hindi_all.jsonl"],

    ("head_qa", "italian"): [PROCESSED_DIR / "head_qa" / "head_qa_italian_train.jsonl"],
    ("head_qa", "spanish"): [PROCESSED_DIR / "head_qa" / "head_qa_spanish_train.jsonl"],

    ("italian_civil_code", "italian"): [PROCESSED_DIR / "italian_civil_code" / "italian_civil_code.jsonl"],
    ("polizia_di_stato", "italian"): [PROCESSED_DIR / "polizia_di_stato" / "polizia_di_stato.jsonl"],
    ("quaderno_prefettizia", "italian"): [PROCESSED_DIR / "quaderno_prefettizia" / "quaderno_prefettizia.jsonl"],
    ("quaderno_prefettizia_economia_politica", "italian"): [
        PROCESSED_DIR / "quaderno_prefettizia" / "quaderno_prefettizia_economia_politica.jsonl"
    ],

    ("mera", "russian"): [PROCESSED_DIR / "mera" / "mera.jsonl"],
    ("rubq", "russian"): [PROCESSED_DIR / "rubq" / "rubq.jsonl"],

    ("spain_boe_legislation", "spanish"): [
        PROCESSED_DIR / "spain_boe_legislation" / "all_spain_boe_legislation.jsonl"
    ],
    ("ccse", "spanish"): [PROCESSED_DIR / "ccse" / "ccse.jsonl"],
}

TARGET_COUNTS = {
    "russian": {
        "include": 100,
        "global_mmlu": 200,
        "mera": 150,
        "rubq": 100,
        "global_piqa": 50,
    },
    "chinese": {
        "include": 100,
        "global_mmlu": 200,
        "ceval": 100,
        "cmmlu": 50,
        "cultsportqa": 100,
        "global_piqa": 50,
    },
    "hindi": {
        "include": 100,
        "global_mmlu": 200,
        "milu": 100,
        "cultsportqa": 150,
        "global_piqa": 50,
    },
    "italian": {
        "include": 100,
        "global_mmlu": 200,
        "head_qa": 50,
        "italian_civil_code": 75,
        "quaderno_prefettizia": 75,
        "polizia_di_stato": 25,
        "quaderno_prefettizia_economia_politica": 25,
        "global_piqa": 50,
    },
    "spanish": {
        "include": 100,
        "global_mmlu": 200,
        "head_qa": 100,
        "spain_boe_legislation": 100,
        "ccse": 50,
        "global_piqa": 50,
    },
}

STRATIFIED_DATASETS = {"global_mmlu", "mera", "ceval", "milu", "cmmlu", "head_qa"}

TOP_LEVEL_ORDER = [
    "source_id",
    "language",
    "dataset",
    "prompt",
    "answer",
    "id",
    "split",
    "task_type",
    "metadata",
]

METADATA_PRIORITY_ORDER = [
    "benchmark_name",
    "benchmark_version",
    "benchmark_component",
    "benchmark_language",
    "benchmark_source_id",
    "subject",
    "category",
    "regionality",
    "subject_category",
    "cultural_sensitivity_label",
    "source_split",
    "source_idx",
    "source_path",
    "sample_id",
    "qid",
    "name",
    "year",
    "n_answers",
    "n_choices",
    "subset",
    "task",
]

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

SMALL_RANDOM_SEED = 142
MMLU_PROX_LANGUAGES = ["russian", "chinese", "hindi", "italian", "spanish"]
MMLU_PROX_N_BASE_QUESTIONS = 200
MMLU_PROX_RANDOM_SEED = 242


def ensure_processed_data_available() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    snapshot_path = Path(
        snapshot_download(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            allow_patterns=f"{HF_PROCESSED_SUBDIR}/**",
        )
    )

    source_dir = snapshot_path / HF_PROCESSED_SUBDIR
    if not source_dir.exists():
        available = sorted(p.name for p in snapshot_path.iterdir() if p.is_dir())
        raise FileNotFoundError(
            f"Could not find {HF_PROCESSED_SUBDIR!r} in HF snapshot {snapshot_path}. "
            f"Available top-level folders: {available}"
        )

    copied = 0
    for source_path in source_dir.rglob("*"):
        if not source_path.is_file():
            continue

        relative_path = source_path.relative_to(source_dir)
        target_path = PROCESSED_DIR / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if target_path.exists() and target_path.stat().st_size == source_path.stat().st_size:
            continue

        shutil.copy2(source_path, target_path)
        copied += 1

    print(f"Processed data source: hf://datasets/{HF_DATASET_REPO}/{HF_PROCESSED_SUBDIR}")
    print(f"Processed data local:  {PROCESSED_DIR}")
    print(f"Files copied/updated:  {copied}")


def stable_seed(*parts, offset=0):
    text = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return offset + int(digest[:8], 16) % 10_000


def read_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception as exc:
                raise ValueError(f"Bad JSON in {path}:{line_no}: {exc}") from exc
    return records


def write_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def clear_generated_benchmark_files():
    patterns = [
        BENCHMARK_DIR / "benchmark_all.jsonl",
        MANIFEST_PATH,
        BY_LANGUAGE_DIR / "*.jsonl",
        BY_DATASET_DIR / "*.jsonl",
        BENCHMARK_DIR / "small" / "benchmark_all_small.jsonl",
        BENCHMARK_DIR / "small" / "by_language" / "*.jsonl",
        BENCHMARK_DIR / "mmlu_prox" / "mmlu_prox_hard_1000.jsonl",
        BENCHMARK_DIR / "mmlu_prox" / "by_language" / "*.jsonl",
    ]

    for pattern in patterns:
        for path in pattern.parent.glob(pattern.name):
            if path.is_file():
                path.unlink()


def get_source_id(record):
    source_id = record.get("source_id")
    if source_id is None or str(source_id).strip() == "":
        source_id = record.get("id")
    if source_id is None:
        return None
    return str(source_id).strip()


def validate_basic_record(record, require_answer=True):
    record_id = record.get("id") or record.get("source_id")
    assert isinstance(record_id, str) and record_id.strip()
    assert isinstance(record.get("dataset"), str) and record["dataset"].strip()
    assert record.get("language") in {"russian", "chinese", "italian", "spanish", "hindi"}
    assert record.get("task_type") == "mcq"
    assert isinstance(record.get("prompt"), str) and record["prompt"].strip()
    assert isinstance(record.get("metadata"), dict)

    if require_answer:
        answer = str(record.get("answer", "")).strip().upper()
        assert answer in VALID_ANSWER_LETTERS
        assert f"{answer}) " in record["prompt"], record_id


def record_subject(record):
    metadata = record.get("metadata", {})
    return (
        metadata.get("subject")
        or metadata.get("subset")
        or metadata.get("task")
        or metadata.get("source_task")
        or metadata.get("category")
        or metadata.get("dataset_name")
        or record.get("subset")
        or record.get("dataset")
        or "unknown"
    )


def sample_uniform(records, n, seed):
    if len(records) < n:
        raise ValueError(f"Need {n} records but only have {len(records)} candidates.")
    if len(records) == n:
        return list(records)
    rng = random.Random(seed)
    return rng.sample(list(records), n)


def sample_subject_stratified(records, n, seed):
    if len(records) < n:
        raise ValueError(f"Need {n} records but only have {len(records)} candidates.")

    by_subject = defaultdict(list)
    for record in records:
        by_subject[record_subject(record)].append(record)

    rng = random.Random(seed)
    subjects = sorted(by_subject)

    for subject in subjects:
        rng.shuffle(by_subject[subject])

    selected = []
    selected_ids = set()

    while len(selected) < n:
        made_progress = False

        for subject in subjects:
            if len(selected) >= n:
                break

            bucket = by_subject[subject]
            while bucket:
                candidate = bucket.pop()
                source_id = get_source_id(candidate)

                if source_id not in selected_ids:
                    selected.append(candidate)
                    selected_ids.add(source_id)
                    made_progress = True
                    break

        if not made_progress:
            break

    if len(selected) < n:
        raise ValueError(f"Only selected {len(selected)} records, need {n}.")

    return selected


def reorder_metadata(metadata):
    metadata = dict(metadata or {})
    ordered = {}

    for key in METADATA_PRIORITY_ORDER:
        if key in metadata:
            ordered[key] = metadata[key]

    for key in sorted(metadata.keys()):
        if key not in ordered:
            ordered[key] = metadata[key]

    return ordered


def standardize_benchmark_record(record):
    record = dict(record)
    source_id = get_source_id(record)
    assert source_id

    record_id = record.get("id")
    if not isinstance(record_id, str) or not record_id.strip():
        record_id = source_id

    metadata = reorder_metadata(record.get("metadata", {}))
    metadata["benchmark_source_id"] = source_id

    standardized = {
        "source_id": source_id,
        "language": str(record["language"]).strip(),
        "dataset": str(record["dataset"]).strip(),
        "prompt": str(record["prompt"]).strip(),
        "answer": str(record["answer"]).strip().upper(),
        "id": str(record_id).strip(),
        "split": "test",
        "task_type": "mcq",
        "metadata": metadata,
    }

    return {key: standardized[key] for key in TOP_LEVEL_ORDER}


def load_candidate_records():
    missing = []

    for key, paths in CANDIDATE_FILES.items():
        if not paths:
            missing.append((key, None))
            continue

        for path in paths:
            if not path.exists():
                missing.append((key, path))

    if missing:
        lines = ["Missing candidate files:"]
        for key, path in missing:
            lines.append(f"  {key}: {path}")
        raise FileNotFoundError("\n".join(lines))

    candidate_records = {}
    rejected = []

    for key, paths in CANDIDATE_FILES.items():
        expected_dataset, expected_language = key
        records = []

        for path in paths:
            for source_row_idx, record in enumerate(read_jsonl(path)):
                try:
                    validate_basic_record(record, require_answer=False)
                    assert record["language"] == expected_language

                    record = dict(record)

                    if not isinstance(record.get("id"), str) or not record["id"].strip():
                        fallback_id = record.get("source_id")
                        assert isinstance(fallback_id, str) and fallback_id.strip()
                        record["id"] = fallback_id

                    actual_dataset = str(record["dataset"])

                    if expected_dataset == "mera":
                        assert actual_dataset.startswith("mera")
                        record["dataset"] = "mera"
                    elif expected_dataset == "spain_boe_legislation":
                        assert actual_dataset.startswith("spain_boe_")
                        record["dataset"] = "spain_boe_legislation"
                    elif expected_dataset in {
                        "polizia_di_stato",
                        "quaderno_prefettizia",
                        "quaderno_prefettizia_economia_politica",
                    }:
                        assert actual_dataset in {
                            expected_dataset,
                            f"{expected_dataset}_clean",
                        }
                        record["dataset"] = expected_dataset
                    else:
                        assert actual_dataset == expected_dataset

                    answer = str(record.get("answer", "")).strip().upper()
                    prompt = str(record.get("prompt", ""))

                    if answer not in VALID_ANSWER_LETTERS:
                        rejected.append((key, path, source_row_idx, "invalid_answer_letter"))
                        continue

                    if f"{answer}) " not in prompt:
                        rejected.append((key, path, source_row_idx, "answer_option_not_in_prompt"))
                        continue

                    record["answer"] = answer
                    records.append(record)

                except Exception as exc:
                    rejected.append((key, path, source_row_idx, repr(exc)))

        candidate_records[key] = records
        print(f"{key}: {len(records)} valid records")

    print(f"Rejected candidate records: {len(rejected)}")
    return candidate_records


def load_distilled_ids():
    return set(), []


def build_main_benchmark(candidate_records, distilled_ids):
    selected_components = {}

    for language, dataset_counts in TARGET_COUNTS.items():
        assert sum(dataset_counts.values()) == 600, f"{language} target count must sum to 600"

        for dataset, n in dataset_counts.items():
            key = (dataset, language)

            if key not in candidate_records:
                raise KeyError(f"Missing candidate records for {key}")

            records = candidate_records[key]
            seed = stable_seed(dataset, language, offset=RANDOM_SEED)

            if dataset in STRATIFIED_DATASETS:
                selected = sample_subject_stratified(records, n, seed)
            else:
                selected = sample_uniform(records, n, seed)

            selected_components[key] = selected
            print(f"{language:8s} | {dataset:45s} | selected {len(selected):4d} / total {len(records):5d}")

    benchmark_records = []

    for (dataset, language), records in selected_components.items():
        for record in records:
            out = dict(record)

            if not isinstance(out.get("id"), str) or not out["id"].strip():
                out["id"] = get_source_id(out)

            out["split"] = "test"
            metadata = dict(out.get("metadata", {}))
            metadata["benchmark_name"] = "cs552_multilingual_regional_mcqa"
            metadata["benchmark_version"] = "v2_diverse_600_per_language"
            metadata["benchmark_component"] = dataset
            metadata["benchmark_language"] = language
            metadata["benchmark_source_id"] = get_source_id(out)
            out["metadata"] = metadata
            benchmark_records.append(out)

    source_ids = [get_source_id(record) for record in benchmark_records]
    assert len(source_ids) == len(set(source_ids)), "Duplicate source IDs in final benchmark."

    for record in benchmark_records:
        validate_basic_record(record)

    benchmark_records = [standardize_benchmark_record(record) for record in benchmark_records]
    return benchmark_records, selected_components


def build_manifest(benchmark_records, language_paths, dataset_paths, distilled_ids, active_distilled_paths):
    by_language = Counter(record["language"] for record in benchmark_records)
    by_dataset = Counter(record["dataset"] for record in benchmark_records)
    by_language_dataset = Counter((record["language"], record["dataset"]) for record in benchmark_records)
    by_language_answer = Counter((record["language"], record["answer"]) for record in benchmark_records)

    return {
        "benchmark_name": "cs552_multilingual_regional_mcqa",
        "version": "v2_diverse_600_per_language",
        "random_seed": RANDOM_SEED,
        "total_records": len(benchmark_records),
        "records_per_language": 600,
        "target_counts": TARGET_COUNTS,
        "source": {
            "hf_dataset_repo": HF_DATASET_REPO,
            "hf_processed_subdir": HF_PROCESSED_SUBDIR,
            "local_processed_dir": str(PROCESSED_DIR),
        },
        "files": {
            "all": str(ALL_PATH.relative_to(BENCHMARK_DIR)),
            "by_language": {
                language: str(path.relative_to(BENCHMARK_DIR))
                for language, path in sorted(language_paths.items())
            },
            "by_dataset": {
                f"{dataset}_{language}": str(path.relative_to(BENCHMARK_DIR))
                for (dataset, language), path in sorted(dataset_paths.items())
            },
        },
        "counts": {
            "by_language": {k: int(v) for k, v in sorted(by_language.items())},
            "by_dataset": {k: int(v) for k, v in sorted(by_dataset.items())},
            "by_language_dataset": {
                f"{language}/{dataset}": int(n)
                for (language, dataset), n in sorted(by_language_dataset.items())
            },
            "by_language_answer": {
                f"{language}/{answer}": int(n)
                for (language, answer), n in sorted(by_language_answer.items())
            },
        },
        "contamination_check": {
            "checked_against_distilled": False,
            "sft_builder_must_exclude_benchmark_source_ids": True,
        },
        "notes": [
            "EXAMS is intentionally excluded.",
            "DEMRE is intentionally excluded from the benchmark because the dataset is very small and reserved for training.",
            "Main benchmark contains 600 examples per language: 100 INCLUDE, 200 Global MMLU, and 300 diverse regional/project-specific examples.",
            "benchmark_all.jsonl is the canonical evaluation file.",
            "by_language files are exact partitions of benchmark_all.jsonl.",
            "by_dataset files are exact partitions of benchmark_all.jsonl.",
            "Benchmark contamination is handled downstream: SFT builders must exclude all source IDs appearing in benchmark files.",
            "MMLU-ProX hard benchmark is separate from benchmark_all.jsonl.",
        ],
    }


def save_main_benchmark(benchmark_records, selected_components, distilled_ids, active_distilled_paths):
    BY_LANGUAGE_DIR.mkdir(parents=True, exist_ok=True)
    BY_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    write_jsonl(benchmark_records, ALL_PATH)

    language_paths = {}
    for language in sorted(TARGET_COUNTS):
        language_records = [record for record in benchmark_records if record["language"] == language]
        assert len(language_records) == 600

        path = BY_LANGUAGE_DIR / f"benchmark_{language}.jsonl"
        write_jsonl(language_records, path)
        language_paths[language] = path

    dataset_paths = {}
    for (dataset, language), selected in sorted(selected_components.items()):
        component_records = [
            record for record in benchmark_records
            if record["dataset"] == dataset
            and record["language"] == language
            and record["metadata"]["benchmark_component"] == dataset
        ]

        expected_n = TARGET_COUNTS[language][dataset]
        assert len(component_records) == expected_n

        path = BY_DATASET_DIR / f"{dataset}_{language}_eval.jsonl"
        write_jsonl(component_records, path)
        dataset_paths[(dataset, language)] = path

    saved_paths = [ALL_PATH] + sorted(BY_LANGUAGE_DIR.glob("*.jsonl")) + sorted(BY_DATASET_DIR.glob("*.jsonl"))

    all_ids = {get_source_id(record) for record in read_jsonl(ALL_PATH)}
    by_language_ids = set()
    by_dataset_ids = set()

    for path in sorted(BY_LANGUAGE_DIR.glob("*.jsonl")):
        by_language_ids.update(get_source_id(record) for record in read_jsonl(path))

    for path in sorted(BY_DATASET_DIR.glob("*.jsonl")):
        by_dataset_ids.update(get_source_id(record) for record in read_jsonl(path))

    assert all_ids == by_language_ids
    assert all_ids == by_dataset_ids

    for path in saved_paths:
        for record in read_jsonl(path):
            assert list(record.keys()) == TOP_LEVEL_ORDER
            assert record["source_id"] == get_source_id(record)
            assert record["split"] == "test"
            assert record["task_type"] == "mcq"
            assert record["answer"] in VALID_ANSWER_LETTERS
            assert f"{record['answer']}) " in record["prompt"]

    manifest = build_manifest(benchmark_records, language_paths, dataset_paths, distilled_ids, active_distilled_paths)

    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest


def build_small_benchmark(benchmark_records, manifest):
    small_benchmark_dir = BENCHMARK_DIR / "small"
    small_by_language_dir = small_benchmark_dir / "by_language"
    small_benchmark_dir.mkdir(parents=True, exist_ok=True)
    small_by_language_dir.mkdir(parents=True, exist_ok=True)

    small_all_path = small_benchmark_dir / "benchmark_all_small.jsonl"
    small_target_counts = {
        language: {dataset: count // 5 for dataset, count in dataset_counts.items()}
        for language, dataset_counts in TARGET_COUNTS.items()
    }

    assert all(sum(dataset_counts.values()) == 120 for dataset_counts in small_target_counts.values())

    small_records = []

    for language, dataset_counts in small_target_counts.items():
        for dataset, n in dataset_counts.items():
            component_records = [
                record for record in benchmark_records
                if record["language"] == language
                and record["dataset"] == dataset
                and record["metadata"]["benchmark_component"] == dataset
            ]

            assert len(component_records) == TARGET_COUNTS[language][dataset]

            seed = stable_seed(language, dataset, offset=SMALL_RANDOM_SEED)

            if dataset in STRATIFIED_DATASETS:
                selected = sample_subject_stratified(component_records, n, seed)
            else:
                selected = sample_uniform(component_records, n, seed)

            small_records.extend(selected)

    source_ids = [record["source_id"] for record in small_records]
    assert len(source_ids) == len(set(source_ids))
    assert len(small_records) == 600

    full_ids = {record["source_id"] for record in benchmark_records}
    write_jsonl(small_records, small_all_path)

    small_language_paths = {}
    for language in sorted(TARGET_COUNTS):
        language_records = [record for record in small_records if record["language"] == language]
        assert len(language_records) == 120

        path = small_by_language_dir / f"benchmark_{language}_small.jsonl"
        write_jsonl(language_records, path)
        small_language_paths[language] = path

    for path in [small_all_path] + sorted(small_by_language_dir.glob("*.jsonl")):
        loaded = read_jsonl(path)
        source_ids = [record["source_id"] for record in loaded]
        assert len(source_ids) == len(set(source_ids))
        assert set(source_ids).issubset(full_ids)

        for record in loaded:
            assert list(record.keys()) == TOP_LEVEL_ORDER
            validate_basic_record(record)
            assert record["split"] == "test"

    manifest["small_benchmark"] = {
        "total_records": len(small_records),
        "records_per_language": 120,
        "random_seed": SMALL_RANDOM_SEED,
        "target_counts": small_target_counts,
        "files": {
            "all": str(small_all_path.relative_to(BENCHMARK_DIR)),
            "by_language": {
                language: str(path.relative_to(BENCHMARK_DIR))
                for language, path in sorted(small_language_paths.items())
            },
        },
        "notes": [
            "The small benchmark is a strict subset of benchmark_all.jsonl.",
            "It contains 1/5 of each full benchmark component, for 120 examples per language.",
            "It is intended for fast debugging and checkpoint smoke tests.",
        ],
    }

    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def weighted_sample_base_questions(complete_base_groups, n, seed):
    rng = random.Random(seed)
    remaining_base_ids = list(complete_base_groups.keys())
    selected_base_ids = []

    while remaining_base_ids and len(selected_base_ids) < n:
        weights = []

        for base_id in remaining_base_ids:
            lang_records = complete_base_groups[base_id]
            subject = record_subject(lang_records["russian"])
            weights.append(SUBJECT_WEIGHTS.get(subject, 1.0))

        chosen_base_id = rng.choices(remaining_base_ids, weights=weights, k=1)[0]
        selected_base_ids.append(chosen_base_id)
        remaining_base_ids.remove(chosen_base_id)

    assert len(selected_base_ids) == n
    return selected_base_ids


def build_mmlu_prox_hard(manifest):
    mmlu_prox_dir = BENCHMARK_DIR / "mmlu_prox"
    mmlu_prox_by_language_dir = mmlu_prox_dir / "by_language"
    mmlu_prox_dir.mkdir(parents=True, exist_ok=True)
    mmlu_prox_by_language_dir.mkdir(parents=True, exist_ok=True)

    mmlu_prox_processed_dir = PROCESSED_DIR / "mmlu_prox"
    mmlu_prox_all_path = mmlu_prox_dir / "mmlu_prox_hard_1000.jsonl"

    eval_by_language = {}

    for language in MMLU_PROX_LANGUAGES:
        path = mmlu_prox_processed_dir / f"mmlu_prox_{language}_eval.jsonl"
        assert path.exists(), path

        records = read_jsonl(path)

        for record in records:
            validate_basic_record(record)
            assert record["dataset"] == "mmlu_prox"
            assert record["language"] == language
            assert record["split"] == "eval"
            assert "base_question_id" in record["metadata"]

        eval_by_language[language] = records

    records_by_base_id = defaultdict(dict)

    for language, records in eval_by_language.items():
        for record in records:
            base_id = record["metadata"]["base_question_id"]
            records_by_base_id[base_id][language] = record

    complete_base_groups = {
        base_id: lang_records
        for base_id, lang_records in records_by_base_id.items()
        if set(lang_records.keys()) == set(MMLU_PROX_LANGUAGES)
    }

    assert len(complete_base_groups) >= MMLU_PROX_N_BASE_QUESTIONS

    selected_base_ids = weighted_sample_base_questions(
        complete_base_groups=complete_base_groups,
        n=MMLU_PROX_N_BASE_QUESTIONS,
        seed=MMLU_PROX_RANDOM_SEED,
    )

    records = []

    for base_id in selected_base_ids:
        lang_records = complete_base_groups[base_id]

        for language in MMLU_PROX_LANGUAGES:
            record = dict(lang_records[language])

            if not isinstance(record.get("id"), str) or not record["id"].strip():
                record["id"] = get_source_id(record)

            record["split"] = "test"

            metadata = dict(record.get("metadata", {}))
            metadata["benchmark_name"] = "cs552_multilingual_regional_mcqa"
            metadata["benchmark_version"] = "mmlu_prox_hard_v1"
            metadata["benchmark_component"] = "mmlu_prox"
            metadata["benchmark_language"] = language
            metadata["benchmark_source_id"] = get_source_id(record)
            metadata["benchmark_base_question_id"] = base_id
            record["metadata"] = metadata

            records.append(standardize_benchmark_record(record))

    assert len(records) == MMLU_PROX_N_BASE_QUESTIONS * len(MMLU_PROX_LANGUAGES)
    write_jsonl(records, mmlu_prox_all_path)

    language_paths = {}

    for language in MMLU_PROX_LANGUAGES:
        language_records = [record for record in records if record["language"] == language]
        assert len(language_records) == MMLU_PROX_N_BASE_QUESTIONS

        path = mmlu_prox_by_language_dir / f"mmlu_prox_{language}_hard_200.jsonl"
        write_jsonl(language_records, path)
        language_paths[language] = path

    for path in [mmlu_prox_all_path] + sorted(mmlu_prox_by_language_dir.glob("*.jsonl")):
        loaded = read_jsonl(path)
        source_ids = [record["source_id"] for record in loaded]
        assert len(source_ids) == len(set(source_ids))

        for record in loaded:
            assert list(record.keys()) == TOP_LEVEL_ORDER
            validate_basic_record(record)
            assert record["dataset"] == "mmlu_prox"
            assert record["split"] == "test"
            assert "benchmark_base_question_id" in record["metadata"]

    manifest["mmlu_prox_hard_benchmark"] = {
        "total_records": len(records),
        "records_per_language": MMLU_PROX_N_BASE_QUESTIONS,
        "unique_base_questions": MMLU_PROX_N_BASE_QUESTIONS,
        "random_seed": MMLU_PROX_RANDOM_SEED,
        "subject_weights": SUBJECT_WEIGHTS,
        "files": {
            "all": str(mmlu_prox_all_path.relative_to(BENCHMARK_DIR)),
            "by_language": {
                language: str(path.relative_to(BENCHMARK_DIR))
                for language, path in sorted(language_paths.items())
            },
        },
        "notes": [
            "MMLU-ProX hard benchmark is separate from benchmark_all.jsonl.",
            "It is sampled only from the held-out MMLU-ProX eval split.",
            "The same base_question_id values are used across all five languages for fair cross-language comparison.",
            "Sampling is weighted toward regional, cultural, society-oriented, health, and biology subjects.",
            "MMLU-ProX may include variable numbers of answer options, up to A-J.",
        ],
    }

    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def main():
    ensure_processed_data_available()

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    BY_LANGUAGE_DIR.mkdir(parents=True, exist_ok=True)
    BY_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    clear_generated_benchmark_files()

    candidate_records = load_candidate_records()
    distilled_ids, active_distilled_paths = load_distilled_ids()

    benchmark_records, selected_components = build_main_benchmark(
        candidate_records,
        distilled_ids,
    )

    manifest = save_main_benchmark(
        benchmark_records,
        selected_components,
        distilled_ids,
        active_distilled_paths,
    )

    build_small_benchmark(benchmark_records, manifest)

    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    build_mmlu_prox_hard(manifest)

    print(f"Benchmark build complete: {BENCHMARK_DIR}")
    print(f"Main benchmark records: {len(benchmark_records)}")
    print("EXAMS excluded.")
    print("DEMRE excluded.")


if __name__ == "__main__":
    main()
