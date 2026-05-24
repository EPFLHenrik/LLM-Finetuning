"""multilingual/train/build_sft_data.py

Builds SFT JSONL files from all loaders, now with distillation integration.

New in v8:
  - distilled_dir: directory containing *_distilled.jsonl files from distillation.
  - Valid distilled examples → train, formatted as:
        <think>COT</think>\n\n\\boxed{gold}
  - Invalid distilled examples → follow invalid_distilled_policy:
        "drop"         → discard entirely
        "raw_train"    → include as raw no-rationale train example
        "raw_val_test" → include in val/test pool like any other raw example
  - Raw examples with source_ids already covered by valid distilled → excluded from train
    (but may still appear in val/test if the loader produces them for those splits)

Output layout:
    <output_dir>/per_dataset/<dataset>_<language>_<split>.jsonl
    <output_dir>/mixed/train_mixed.jsonl
    <output_dir>/mixed/val_mixed.jsonl
    <output_dir>/mixed/test_mixed.jsonl
    <output_dir>/build_summary.csv

Usage:
    python -m multilingual.train.build_sft_data --config configs/sft/v8.json --overwrite
    python -m multilingual.train.build_sft_data --debug
"""

import argparse
import csv
import hashlib
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import hashlib

from multilingual.config import REPO, SYSTEM_PROMPT, TARGET_LANGUAGES
from multilingual.loaders.belebele_loader import load_belebele_examples
from multilingual.loaders.ceval_loader import load_ceval_examples
from multilingual.loaders.cultsportqa_loader import load_cultsportqa_examples
from multilingual.loaders.exams_loader import load_exams_examples
from multilingual.loaders.head_qa_loader import load_head_qa_examples
from multilingual.loaders.include_loader import load_include_recarved, load_include_examples
from multilingual.loaders.mera_loader import load_mera_examples
from multilingual.loaders.mgsm_loader import load_mgsm_examples
from multilingual.loaders.milu_loader import load_milu_examples
from multilingual.loaders.mmmlu_loader import load_mmmlu_examples
from multilingual.loaders.xcopa_loader import load_xcopa_examples
from multilingual.loaders.global_mmlu_loader import load_global_mmlu_examples
from multilingual.loaders.global_piqa_loader import load_global_piqa_recarved, load_global_piqa_examples

# ---------------------------------------------------------------------------
# Global defaults
# ---------------------------------------------------------------------------

DEFAULT_MAX_PER_LANGUAGE           = 1000
DEFAULT_MAX_PER_DATASET            = 5000
DEFAULT_MAX_PER_LANGUAGE_MIXED     = 2500
DEFAULT_MAX_PER_LANGUAGE_MIXED_VAL = 1000
DEFAULT_SFT_BASE_DIR = REPO / "multilingual" / "data" / "sft"
DEBUG_MAX            = 20
SHUFFLE_SEED         = 42
ALL_SPLITS           = ("train", "val", "test")

# Default policy for invalid distilled examples
DEFAULT_INVALID_DISTILLED_POLICY = "raw_train"  # drop | raw_train | raw_val_test

# ---------------------------------------------------------------------------
# Dataset registry  (unchanged from v7)
# ---------------------------------------------------------------------------

DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "include": {
        "languages":  None,
        "mode":       "recarve",
        "split_caps": {"train": 0, "val": None, "test": None},
    },
    "global_piqa": {
        "languages":  None,
        "mode":       "recarve",
        "split_caps": {"train": 0, "val": None, "test": None},
        "max_per_dataset": 0,
    },
    "mgsm": {
        "languages":        None,
        "max_per_language": 50,
        "max_per_dataset":  250,
        "split_caps":       {"val": 100},
    },
    "mmmlu": {
        "languages":       ["spanish", "hindi", "italian", "chinese"],
        "max_per_dataset": 0,
    },
    "milu": {
        "languages":        ["hindi"],
        "max_per_language": 2000,
        "split_caps":       {"val": 300},
    },
    "ceval": {
        "languages":     ["chinese"],
        "language_caps": {"chinese": 2000},
        "split_caps":    {"val": 300},
    },
    "cultsportqa": {
        "languages":        ["chinese", "hindi"],
        "max_per_language": 1000,
        "max_per_dataset":  2000,
        "split_caps":       {"val": 400},
    },
    "belebele": {
        "languages":        None,
        "max_per_language": 300,
        "max_per_dataset":  1500,
        "split_caps":       {"val": 100},
    },
    "xcopa": {
        "languages":       ["italian", "chinese"],
        "max_per_dataset": 0,
    },
    "exams": {
        "languages":  ["italian", "spanish"],
        "split_caps": {"val": 300},
    },
    "head_qa": {
        "languages":     ["spanish", "italian"],
        "language_caps": {"spanish": 1000, "italian": 400},
        "split_caps":    {"val": 300},
    },
    "mera": {
        "languages":        ["russian"],
        "max_per_language": 2000,
        "max_per_dataset":  2000,
        "stratify_by":      "subset",
        "subject_caps": {
            "mamuramu":     285,
            "ruopenbookqa": 600,
            "ruworldtree":  115,
            "mathlogicqa":  150,
            "parus":          0,
            "use":            0,
            "rcb":            0,
            "rwsd":           0,
        },
        "split_caps": {"val": 200},
    },
    "global_mmlu": {
        "languages":     None,
        "language_caps": {
            "russian": 2200,
            "chinese": 2200,
            "hindi":    600,
            "italian":  300,
            "spanish":  300,
        },
        "split_caps": {"val": 600},
    },
}

# ---------------------------------------------------------------------------
# Config helpers  (unchanged)
# ---------------------------------------------------------------------------

def _dataset_cfg(dataset: str) -> Dict[str, Any]:
    return DATASET_REGISTRY.get(dataset, {})


def _languages(dataset: str) -> List[str]:
    cfg   = _dataset_cfg(dataset)
    langs = cfg.get("languages")
    return langs if langs is not None else list(TARGET_LANGUAGES.keys())


def _max_per_language(dataset: str, language: str, global_cap: int) -> int:
    cfg       = _dataset_cfg(dataset)
    lang_caps = cfg.get("language_caps", {})
    if language in lang_caps:
        v = lang_caps[language]
        return v if v is not None else 10_000_000
    v = cfg.get("max_per_language", global_cap)
    return v if v is not None else 10_000_000


def _split_cap(dataset: str, split: str, global_cap: int) -> int:
    cfg        = _dataset_cfg(dataset)
    split_caps = cfg.get("split_caps", {})
    if split in split_caps:
        v = split_caps[split]
        return v if v is not None else 10_000_000
    if split == "test":
        return 0
    v = cfg.get("max_per_dataset", global_cap)
    return v if v is not None else 10_000_000


def _split_enabled(dataset: str, split: str) -> bool:
    return _split_cap(dataset, split, 10_000_000) != 0


def _is_recarve(dataset: str) -> bool:
    return _dataset_cfg(dataset).get("mode") == "recarve"

# ---------------------------------------------------------------------------
# COT extraction from distilled teacher output
# ---------------------------------------------------------------------------

def split_distilled_answer(clean: str, gold: str) -> Tuple[str, str]:
    """
    Extract (cot, final_answer) from a teacher's clean output.

    The teacher output looks like:
        Explanation sentence(s).

        \\boxed{A}

    We strip trailing lines containing \\boxed{} to get the COT,
    then always use gold as the authoritative final answer.

    Returns:
        cot:          explanation text without any \\boxed{} on the final line
        final_answer: f"\\boxed{{{gold}}}"
    """
    lines = clean.strip().splitlines()

    # Remove trailing empty lines
    while lines and not lines[-1].strip():
        lines.pop()

    # Remove trailing line(s) that contain \\boxed{}
    while lines and "\\boxed{" in lines[-1]:
        lines.pop()
        # Also remove newly trailing empty lines
        while lines and not lines[-1].strip():
            lines.pop()

    cot = "\n".join(lines).strip()
    final_answer = f"\\boxed{{{gold}}}"
    return cot, final_answer


def _make_distilled_assistant_message(cot: str, gold: str) -> str:
    """Build the assistant message for a valid distilled example."""
    answer = gold.strip().upper()
    boxed  = f"\\boxed{{{answer}}}"
    if cot:
        return f"<think>\n{cot}\n</think>\n\n{boxed}"
    # Empty COT — still include think tags for format consistency
    return f"<think>\n</think>\n\n{boxed}"


def _make_raw_assistant_message(answer: str) -> str:
    """Build the assistant message for a raw (non-distilled) example."""
    answer = answer.strip().upper()
    return f"\\boxed{{{answer}}}"

# ---------------------------------------------------------------------------
# SFT record helpers
# ---------------------------------------------------------------------------

def _hash_source_id(ex: Dict[str, Any]) -> str:
    raw = "\n".join([
        str(ex["dataset"]), str(ex["language"]),
        str(ex["task_type"]), str(ex["prompt"]).strip(),
    ])
    return f"{ex['dataset']}_{ex['language']}_{hashlib.sha1(raw.encode()).hexdigest()[:16]}"


# def _make_assistant_message(ex: Dict[str, Any]) -> str:
#     """Build assistant message for a raw (non-distilled) example."""
#     answer = str(ex["answer"]).strip()
#     if ex["task_type"] == "mcq":
#         answer = answer.upper()
#     boxed = f"\\boxed{{{answer}}}"
#     cot = ex.get("cot") or ex.get("rationale") or ex.get("reasoning")
#     if cot:
#         cot = str(cot).strip()
#         return f"<think>\n{cot}\n</think>\n\n{boxed}"
#     return boxed

def _make_assistant_message(ex: Dict[str, Any]) -> str:
    answer = str(ex["answer"]).strip()
    if ex["task_type"] == "mcq":
        answer = answer.upper()
    boxed = f"\\boxed{{{answer}}}"
    cot = ex.get("cot") or ex.get("rationale") or ex.get("reasoning")
    if cot:
        return f"<think>\n{str(cot).strip()}\n</think>\n\n{boxed}"
    # always include empty think tags for format consistency
    return f"<think>\n</think>\n\n{boxed}"


def _to_sft_record(ex: Dict[str, Any], split: str, idx: int) -> Dict[str, Any]:
    source_id = str(ex.get("source_id") or ex.get("id") or _hash_source_id(ex))
    loader_id = str(ex.get("id", source_id))

    return {
        "id":        f"{ex['dataset']}_{split}_{ex['language']}_{idx}",
        "source_id": source_id,
        "loader_id": loader_id,
        "dataset":   ex["dataset"],
        "language":  ex["language"],
        "split":     split,
        "task_type": ex["task_type"],
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": ex["prompt"]},
            {"role": "assistant", "content": _make_assistant_message(ex)},
        ],
        "answer":   ex["answer"],
        "metadata": ex.get("metadata", {}),
    }

def _to_records(examples: List[Dict], split: str) -> List[Dict]:
    return [_to_sft_record(ex, split, i) for i, ex in enumerate(examples)]


def _dedupe(records: List[Dict]) -> List[Dict]:
    seen, out = set(), []
    for r in records:
        if r["source_id"] not in seen:
            seen.add(r["source_id"])
            out.append(r)
    return out


def _remove_train_overlap(train: List[Dict], others: List[List[Dict]]) -> List[Dict]:
    other_ids = {r["source_id"] for split in others for r in split}
    return [r for r in train if r["source_id"] not in other_ids]


def _reindex(records: List[Dict], split: str) -> List[Dict]:
    for i, r in enumerate(records):
        r["id"]    = f"{r['dataset']}_{split}_{r['language']}_{i}"
        r["split"] = split
    return records


def _write_jsonl(records: List[Dict], path: Path, overwrite: bool) -> bool:
    if path.exists() and not overwrite:
        print(f"  [SKIP] exists: {path}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records):>5} records → {path.name}")
    return True


def _load_jsonl(path: Path) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

# ---------------------------------------------------------------------------
# Distillation loading
# ---------------------------------------------------------------------------

def _normalize_for_dedupe(x: str) -> str:
    return re.sub(r"\s+", " ", str(x).strip())


def _distilled_dedupe_key(r: Dict[str, Any]) -> str:
    prompt = ""
    if r.get("messages") and len(r["messages"]) >= 2:
        prompt = r["messages"][1]["content"]

    raw = "\n".join([
        str(r.get("dataset", "")),
        str(r.get("language", "")),
        str(r.get("task_type", "")),
        _normalize_for_dedupe(prompt),
        str(r.get("answer", "")).strip().upper(),
    ])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

def _load_distilled_dirs(
    distilled_dirs: List[Path],
) -> Tuple[List[Dict], Set[str]]:
    """
    Scan distilled dirs for *_distilled.jsonl files.
    Returns:
        distilled_train_records: SFT records ready for train (valid examples only)
        distilled_source_ids:    source_ids of ALL attempted examples (valid + invalid)
                                 used to exclude from raw train pool
    """
    distilled_train_records: List[Dict] = []
    distilled_source_ids:    Set[str]   = set()

    for distilled_dir in distilled_dirs:
        if not distilled_dir.exists():
            print(f"  [WARN] distilled_dir not found: {distilled_dir}")
            continue

        # Load all *_distilled.jsonl files (valid examples only)
        for path in sorted(distilled_dir.glob("*_distilled.jsonl")):
            records = _load_jsonl(path)
            print(f"  Loading distilled: {path.name} ({len(records)} valid)")
            for r in records:
                gold      = str(r.get("answer", r.get("distill", {}).get("gold", ""))).strip().upper()
                source_id = str(r.get("source_id", ""))
                dataset   = str(r.get("dataset", "unknown"))
                language  = str(r.get("language", "unknown"))
                prompt    = str(r.get("prompt", ""))

                # If prompt missing, try to reconstruct from messages
                if not prompt and r.get("messages") and len(r["messages"]) >= 2:
                    prompt = str(r["messages"][1]["content"])

                clean = r.get("distill", {}).get("clean", "")

                if clean:
                    cot, _ = split_distilled_answer(clean, gold)
                else:
                    cot = ""

                assistant_content = _make_distilled_assistant_message(cot, gold)

                sft_record = {
                    "id":        f"distilled_{dataset}_{language}_{len(distilled_train_records)}",
                    "source_id": source_id,
                    "loader_id": str(r.get("id", source_id)),
                    "dataset":   dataset,
                    "language":  language,
                    "split":     "train",
                    "task_type": str(r.get("task_type", "mcq")),
                    "messages": [
                        {"role": "system",    "content": SYSTEM_PROMPT},
                        {"role": "user",      "content": prompt},
                        {"role": "assistant", "content": assistant_content},
                    ],
                    "answer":   gold,
                    "metadata": {
                        **r.get("metadata", {}),
                        "distilled":           True,
                        "used_teacher":        r.get("distill", {}).get("used_teacher"),
                        "distill_source_file": path.name,
                    },
                }
                distilled_train_records.append(sft_record)

        # Load all *_all.jsonl files to collect ALL attempted source_ids
        # (so we can exclude them from the raw train pool)
        # If no *_all.jsonl files exist, fall back to using distilled source_ids
        all_files = [p for p in sorted(distilled_dir.glob("*_all.jsonl"))
                     if "_shuffle_meta" not in p.name]
        if all_files:
            for path in all_files:
                records = _load_jsonl(path)
                for r in records:
                    sid = str(r.get("source_id", ""))
                    if sid:
                        distilled_source_ids.add(sid)
        else:
            # Option B fallback: only exclude source_ids of valid distilled examples
            for r in distilled_train_records:
                sid = str(r.get("source_id", ""))
                if sid:
                    distilled_source_ids.add(sid)
            print(f"  [INFO] No *_all.jsonl found in {distilled_dir} — "
                  f"using distilled source_ids only for exclusion")

    # dedupe valid distilled records by dataset + language + task_type + prompt + answer
    seen: Set[str] = set()
    deduped = []
    conflicts: Dict[str, Set[str]] = defaultdict(set)

    for r in distilled_train_records:
        prompt = ""
        if r.get("messages") and len(r["messages"]) >= 2:
            prompt = r["messages"][1]["content"]
        prompt_key_raw = "\n".join([
            str(r.get("dataset", "")),
            str(r.get("language", "")),
            str(r.get("task_type", "")),
            _normalize_for_dedupe(prompt),
        ])
        prompt_key = hashlib.sha1(prompt_key_raw.encode("utf-8")).hexdigest()
        conflicts[prompt_key].add(str(r.get("answer", "")).strip().upper())
        key = _distilled_dedupe_key(r)
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    n_conflicts = sum(1 for answers in conflicts.values() if len(answers) > 1)
    if n_conflicts:
        print(f"  [WARN] Found {n_conflicts} prompt duplicates with conflicting answers")
    removed = len(distilled_train_records) - len(deduped)
    if removed:
        print(f"  Removed {removed} duplicate prompt+answer records across distilled files")
    distilled_train_records = deduped
    
    print(f"  Distilled: {len(distilled_train_records)} valid train records")
    print(f"  Distilled: {len(distilled_source_ids)} total attempted source_ids excluded from raw train")
    return distilled_train_records, distilled_source_ids

# ---------------------------------------------------------------------------
# Loader dispatch  (unchanged from v7)
# ---------------------------------------------------------------------------

def _load_examples(
    dataset: str, language: str, split: str,
    n: Optional[int], hf_token: str,
) -> List[Dict[str, Any]]:
    mode = str(_dataset_cfg(dataset).get("mode", "as_is"))
    if dataset == "include":
        return load_include_examples(language=language, n=n, split=split, mode=mode)
    if dataset == "global_piqa":
        return load_global_piqa_examples(language=language, n=n, split=split, mode=mode)
    if dataset == "mgsm":
        return load_mgsm_examples(language=language, n=n, split=split)
    if dataset == "mmmlu":
        return load_mmmlu_examples(language=language, n=n, split=split)
    if dataset == "milu":
        return load_milu_examples(n=n, split=split, hf_token=hf_token)
    if dataset == "ceval":
        return load_ceval_examples(n=n, split=split)
    if dataset == "cultsportqa":
        return load_cultsportqa_examples(language=language, n=n, split=split)
    if dataset == "belebele":
        return load_belebele_examples(language=language, n=n, split=split)
    if dataset == "xcopa":
        return load_xcopa_examples(language=language, n=n, split=split)
    if dataset == "exams":
        return load_exams_examples(language=language, n=n, split=split)
    if dataset == "head_qa":
        return load_head_qa_examples(language=language, n=n, split=split)
    if dataset == "mera":
        return load_mera_examples(language=language, n=n, split=split)
    if dataset == "global_mmlu":
        return load_global_mmlu_examples(language=language, n=n, split=split)
    raise ValueError(f"Unknown dataset: {dataset}")


def _load_recarved(
    dataset: str, language: str,
    n_val: Optional[int], n_test: Optional[int],
) -> Dict[str, List[Dict]]:
    if dataset == "include":
        val_ex, test_ex = load_include_recarved(language, n_val=n_val, n_test=n_test)
    elif dataset == "global_piqa":
        val_ex, test_ex = load_global_piqa_recarved(language, n_val=n_val, n_test=n_test)
    else:
        raise ValueError(f"Recarve not implemented for dataset: {dataset}")
    result = {}
    if val_ex:
        result["val"]  = _dedupe(_to_records(val_ex,  "val"))
    if test_ex:
        result["test"] = _dedupe(_to_records(test_ex, "test"))
    return result

# ---------------------------------------------------------------------------
# Stratified sampling  (unchanged)
# ---------------------------------------------------------------------------

def _stratified_sample(
    records: List[Dict],
    stratify_by: str,
    subject_caps: Dict[str, Optional[int]],
    global_cap: int,
    rng: random.Random,
) -> List[Dict]:
    by_group: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        key = str(r.get("metadata", {}).get(stratify_by, "unknown"))
        by_group[key].append(r)
    for g in by_group.values():
        rng.shuffle(g)
    result, used, uncapped = [], 0, []
    for group, items in by_group.items():
        if group in subject_caps:
            cap = subject_caps[group]
            if cap == 0:
                continue
            take = items if cap is None else items[:cap]
            result.extend(take)
            used += len(take)
        else:
            uncapped.append((group, items))
    remaining = max(0, global_cap - used)
    if uncapped and remaining > 0:
        per_group = max(1, remaining // len(uncapped))
        for _, items in uncapped:
            result.extend(items[:per_group])
    rng.shuffle(result)
    return result[:global_cap]

# ---------------------------------------------------------------------------
# Mixed file builder  (unchanged)
# ---------------------------------------------------------------------------

def _build_mixed(
    per_dir: Path,
    split: str,
    global_max_per_language: int,
    global_max_per_dataset: int,
    max_per_language_mixed: Optional[int],
    rng: random.Random,
    debug: bool,
) -> List[Dict]:
    mixed: List[Dict] = []
    for dataset in DATASET_REGISTRY:
        if not _split_enabled(dataset, split):
            continue
        cfg          = _dataset_cfg(dataset)
        stratify_by  = cfg.get("stratify_by")
        subject_caps = cfg.get("subject_caps", {})
        dataset_pool: List[Dict] = []
        for language in _languages(dataset):
            path = per_dir / f"{dataset}_{language}_{split}.jsonl"
            if not path.exists():
                continue
            records = _load_jsonl(path)
            if not records:
                continue
            rng.shuffle(records)
            if debug:
                dataset_pool.extend(records)
                continue
            cap = _max_per_language(dataset, language, global_max_per_language)
            if stratify_by:
                sampled = _stratified_sample(records, stratify_by, subject_caps, cap, rng)
            else:
                sampled = records[:cap]
            dataset_pool.extend(sampled)
        if not dataset_pool:
            continue
        if not debug:
            ds_cap = _split_cap(dataset, split, global_max_per_dataset)
            rng.shuffle(dataset_pool)
            dataset_pool = dataset_pool[:ds_cap]
        mixed.extend(dataset_pool)

    if not debug and max_per_language_mixed is not None:
        by_lang: Dict[str, List[Dict]] = defaultdict(list)
        for r in mixed:
            by_lang[r["language"]].append(r)
        balanced: List[Dict] = []
        for lang, records in by_lang.items():
            rng.shuffle(records)
            balanced.extend(records[:max_per_language_mixed])
        mixed = balanced

    rng.shuffle(mixed)
    return mixed


def _dataset_total_cap_for_distilled(dataset: str, global_cap: int) -> int:
    """
    Cap distilled examples by dataset without using split_caps.

    Some datasets have split_caps["train"] = 0 because their raw loader is
    evaluation-only/recarved, but distilled examples from those datasets should
    still be allowed in train. Therefore distilled caps use max_per_dataset only.
    """
    cfg = _dataset_cfg(dataset)
    v = cfg.get("max_per_dataset", global_cap)
    return v if v is not None else 10_000_000


def _cap_distilled_records(
    records: List[Dict],
    global_max_per_language: int,
    global_max_per_dataset: int,
    rng: random.Random,
    debug: bool,
) -> List[Dict]:
    """
    Apply the same dataset/language caps to distilled records before they are
    written and added to the mixed train set.

    This prevents large distilled sources, especially CEval and MILU, from
    bypassing the normal sampling caps.
    """
    if debug:
        return records

    by_dataset_language: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for r in records:
        by_dataset_language[(r["dataset"], r["language"])].append(r)

    by_dataset: Dict[str, List[Dict]] = defaultdict(list)

    for (dataset, language), items in by_dataset_language.items():
        rng.shuffle(items)
        cap = _max_per_language(dataset, language, global_max_per_language)
        by_dataset[dataset].extend(items[:cap])

    capped: List[Dict] = []
    for dataset, items in by_dataset.items():
        rng.shuffle(items)
        ds_cap = _dataset_total_cap_for_distilled(dataset, global_max_per_dataset)
        capped.extend(items[:ds_cap])

    rng.shuffle(capped)
    return capped


def _cap_records_by_language(
    records: List[Dict],
    max_per_language: Optional[int],
    rng: random.Random,
    debug: bool,
) -> List[Dict]:
    """
    Final language cap applied after adding distilled records to the train mix.
    This is needed because distilled records are appended after _build_mixed().
    """
    if debug or max_per_language is None:
        return records

    by_lang: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        by_lang[r["language"]].append(r)

    capped: List[Dict] = []
    for _, items in by_lang.items():
        rng.shuffle(items)
        capped.extend(items[:max_per_language])

    rng.shuffle(capped)
    return capped

# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build(
    output_dir: Path,
    max_per_language: int,
    max_per_dataset: int,
    max_per_language_mixed: Optional[int],
    debug: bool,
    overwrite: bool,
    hf_token: str,
    distilled_dirs: Optional[List[Path]] = None,
    invalid_distilled_policy: str = DEFAULT_INVALID_DISTILLED_POLICY,
) -> None:
    per_dir   = output_dir / "per_dataset"
    mixed_dir = output_dir / "mixed"
    rng       = random.Random(SHUFFLE_SEED)
    summary_rows: List[Dict] = []

    # ── Step 0: load distilled examples ──────────────────────────────────────
    distilled_train: List[Dict] = []
    distilled_ids:   Set[str]   = set()

    if distilled_dirs:
        print("\n── Loading distilled examples ──")
        distilled_train, distilled_ids = _load_distilled_dirs(distilled_dirs)
        before_distilled_cap = len(distilled_train)
        distilled_train = _cap_distilled_records(
            distilled_train,
            global_max_per_language=max_per_language,
            global_max_per_dataset=max_per_dataset,
            rng=rng,
            debug=debug,
        )
        removed_distilled = before_distilled_cap - len(distilled_train)
        if removed_distilled:
            print(f"  Capped distilled train examples: {before_distilled_cap} → {len(distilled_train)}")
        print(f"  → {len(distilled_train)} distilled train examples loaded")

    # ── Step 1: per-dataset/language files ───────────────────────────────────
    for dataset in DATASET_REGISTRY:
        for language in _languages(dataset):
            print(f"\n── {dataset}/{language} ──")

            if _is_recarve(dataset):
                n_val  = _split_cap(dataset, "val",  max_per_dataset)
                n_test = _split_cap(dataset, "test", max_per_dataset)
                n_val  = None if n_val  >= 10_000_000 else n_val
                n_test = None if n_test >= 10_000_000 else n_test
                try:
                    split_records = _load_recarved(dataset, language, n_val, n_test)
                except Exception as e:
                    print(f"  [SKIP recarve] {e}")
                    continue
            else:
                active = [s for s in ALL_SPLITS if _split_enabled(dataset, s)]
                raw: Dict[str, List[Dict]] = {}

                for split in active:
                    n_cap = DEBUG_MAX if debug else None
                    print(f"  Loading {split} ...")
                    try:
                        examples = _load_examples(dataset, language, split, n_cap, hf_token)
                    except Exception as e:
                        print(f"  [SKIP] {e}")
                        raw[split] = []
                        continue
                    if not examples:
                        print(f"  [SKIP] empty")
                        raw[split] = []
                        continue
                    records = _dedupe(_to_records(examples, split))
                    raw[split] = records

                # Exclude distilled source_ids from train pool only
                if raw.get("train") and distilled_ids:
                    before = len(raw["train"])
                    raw["train"] = [r for r in raw["train"]
                                    if r["source_id"] not in distilled_ids]
                    excluded = before - len(raw["train"])
                    if excluded:
                        print(f"  Excluded {excluded} train examples already covered by distillation")

                # Remove train overlap with val + test
                if raw.get("train"):
                    others = [v for k, v in raw.items() if k != "train" and v]
                    before = len(raw["train"])
                    raw["train"] = _remove_train_overlap(raw["train"], others)
                    removed = before - len(raw["train"])
                    if removed:
                        print(f"  Removed {removed} train/eval overlaps")

                split_records = {k: v for k, v in raw.items() if v}

            for split, records in split_records.items():
                records = _reindex(records, split)
                out_path = per_dir / f"{dataset}_{language}_{split}.jsonl"
                if _write_jsonl(records, out_path, overwrite):
                    summary_rows.append({
                        "dataset": dataset, "language": language,
                        "split": split, "n": len(records), "path": str(out_path),
                    })

    # ── Step 2: write distilled per-language train files ─────────────────────
    if distilled_train:
        by_lang_dataset: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
        for r in distilled_train:
            by_lang_dataset[(r["dataset"], r["language"])].append(r)

        for (dataset, language), records in sorted(by_lang_dataset.items()):
            records = _reindex(records, "train")
            out_path = per_dir / f"{dataset}_{language}_distilled_train.jsonl"
            if _write_jsonl(records, out_path, overwrite):
                summary_rows.append({
                    "dataset": f"{dataset}[distilled]", "language": language,
                    "split": "train", "n": len(records), "path": str(out_path),
                })

    # ── Step 3: mixed files ───────────────────────────────────────────────────
    produced_splits = set()
    for dataset in DATASET_REGISTRY:
        for split in ALL_SPLITS:
            if _split_enabled(dataset, split):
                produced_splits.add(split)
    if distilled_train:
        produced_splits.add("train")

    for split in ALL_SPLITS:
        if split not in produced_splits:
            continue
        if split == "train":
            lang_cap = max_per_language_mixed
        elif split == "val":
            lang_cap = DEFAULT_MAX_PER_LANGUAGE_MIXED_VAL
        else:
            lang_cap = None

        mixed = _build_mixed(
            per_dir=per_dir,
            split=split,
            global_max_per_language=max_per_language,
            global_max_per_dataset=max_per_dataset,
            max_per_language_mixed=lang_cap,
            rng=rng,
            debug=debug,
        )

        # Add distilled examples to train mixed. They were already capped by
        # dataset/language above, then the final mixed language cap is applied
        # again after appending them.
        if split == "train" and distilled_train:
            mixed = mixed + distilled_train
            rng.shuffle(mixed)
            print(f"  Added {len(distilled_train)} distilled examples to train mixed")

        if split == "train" and not debug and max_per_language_mixed is not None:
            before_final_lang_cap = len(mixed)
            mixed = _cap_records_by_language(
                mixed,
                max_per_language=max_per_language_mixed,
                rng=rng,
                debug=debug,
            )
            removed_final_lang_cap = before_final_lang_cap - len(mixed)
            if removed_final_lang_cap:
                print(
                    f"  Applied final train language cap: "
                    f"{before_final_lang_cap} → {len(mixed)}"
                )

        if not mixed:
            continue
        mixed_path = mixed_dir / f"{split}_mixed.jsonl"
        if _write_jsonl(mixed, mixed_path, overwrite):
            summary_rows.append({
                "dataset": "mixed", "language": "all",
                "split": split, "n": len(mixed), "path": str(mixed_path),
            })

    # ── Step 4: summary CSV ──────────────────────────────────────────────────
    summary_path = output_dir / "build_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "language", "split", "n", "path"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSummary: {summary_path}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_per_language",       type=int, default=DEFAULT_MAX_PER_LANGUAGE)
    parser.add_argument("--max_per_dataset",        type=int, default=DEFAULT_MAX_PER_DATASET)
    parser.add_argument("--max_per_language_mixed", type=int, default=DEFAULT_MAX_PER_LANGUAGE_MIXED)
    parser.add_argument("--debug",     action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--config",    type=Path, default=None,
                        help="JSON config overriding DATASET_REGISTRY and global caps.")
    parser.add_argument("--distilled_dirs", nargs="*", type=Path, default=None,
                        help="Directories containing *_distilled.jsonl files.")
    parser.add_argument("--invalid_distilled_policy", type=str,
                        default=DEFAULT_INVALID_DISTILLED_POLICY,
                        choices=["drop", "raw_train", "raw_val_test"],
                        help="What to do with invalid distilled examples.")
    return parser.parse_args()


def main():
    args = parse_args()

    max_per_language = args.max_per_language
    max_per_dataset  = args.max_per_dataset
    max_mixed        = args.max_per_language_mixed if args.max_per_language_mixed > 0 else None
    output_dir       = DEFAULT_SFT_BASE_DIR
    distilled_dirs   = args.distilled_dirs or []
    invalid_policy   = args.invalid_distilled_policy

    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        max_per_language = cfg.get("max_per_language", max_per_language)
        max_per_dataset  = cfg.get("max_per_dataset",  max_per_dataset)
        max_mixed        = cfg.get("max_per_language_mixed", max_mixed)
        invalid_policy   = cfg.get("invalid_distilled_policy", invalid_policy)
        if output_name := cfg.get("output_name"):
            output_dir = DEFAULT_SFT_BASE_DIR / output_name
        if "dataset_registry" in cfg:
            DATASET_REGISTRY.update(cfg["dataset_registry"])
        if "distilled_dirs" in cfg:
            distilled_dirs = [Path(d) for d in cfg["distilled_dirs"]]

    hf_token = os.environ.get("HF_TOKEN", "")
    print(f"Debug:      {args.debug}  Overwrite: {args.overwrite}")
    print(f"Config:     {args.config or 'default'}")
    print(f"Output dir: {output_dir}")
    print(f"Caps:       max_per_language={max_per_language}, "
          f"max_per_dataset={max_per_dataset}, "
          f"max_per_language_mixed={max_mixed}")
    if distilled_dirs:
        print(f"Distilled:  {[str(d) for d in distilled_dirs]}")
        print(f"Invalid policy: {invalid_policy}")
    print()

    build(
        output_dir=output_dir,
        max_per_language=max_per_language,
        max_per_dataset=max_per_dataset,
        max_per_language_mixed=max_mixed,
        debug=args.debug,
        overwrite=args.overwrite,
        hf_token=hf_token,
        distilled_dirs=distilled_dirs or None,
        invalid_distilled_policy=invalid_policy,
    )


if __name__ == "__main__":
    main()