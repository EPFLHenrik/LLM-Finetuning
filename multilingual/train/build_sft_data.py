"""multilingual/train/build_sft_data.py

Builds SFT JSONL files from all loaders, with distillation integration.

New in v10:
  - Distilled examples are split into train (1 - DISTILLED_VAL_FRACTION)
    and val (DISTILLED_VAL_FRACTION) per dataset/language group.
  - This ensures val loss reflects the same CoT format as train,
    fixing the train/val distribution mismatch from v8/v9.
  - Distilled val examples are written to per_dataset/ and picked up
    by _build_mixed for the val mixed file automatically.
  - All other logic (raw loaders, caps, recarve, stratified sampling)
    is unchanged.

Output layout:
    <output_dir>/per_dataset/<dataset>_<language>_<split>.jsonl
    <output_dir>/per_dataset/<dataset>_<language>_distilled_train.jsonl
    <output_dir>/per_dataset/<dataset>_<language>_distilled_val.jsonl
    <output_dir>/mixed/train_mixed.jsonl
    <output_dir>/mixed/val_mixed.jsonl
    <output_dir>/mixed/include_val.jsonl   ← INCLUDE-only val subset
    <output_dir>/build_summary.csv

Usage:
    python -m multilingual.train.build_sft_data --config configs/sft/v10.json --overwrite
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

from multilingual.config import REPO, SYSTEM_PROMPT, TARGET_LANGUAGES, VAL_FRACTION, VAL_SEED
from multilingual.loaders.belebele_loader import load_belebele_examples
from multilingual.loaders.ceval_loader import load_ceval_examples
from multilingual.loaders.cultsportqa_loader import load_cultsportqa_examples
from multilingual.loaders.exams_loader import load_exams_examples
from multilingual.loaders.head_qa_loader import load_head_qa_examples
from multilingual.loaders.include_loader import load_include_recarved, load_include_examples
from multilingual.loaders.mera_loader import load_mera_examples
from multilingual.loaders.mgsm_loader import load_mgsm_examples
from multilingual.loaders.milu_loader import load_milu_examples
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
ALL_SPLITS           = ("train", "val", "test")

DEFAULT_INVALID_DISTILLED_POLICY = "raw_train"

# Fraction of distilled examples held out for val per dataset/language group.
# This ensures val loss reflects the same CoT format as train.
DISTILLED_VAL_FRACTION = 0.20

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "include": {
        "languages":  None,
        "mode":       "recarve",
        "split_caps": {"train": 0, "val": None, "test": None},
    },
    "global_piqa": {
        "languages": None,
        "mode": "recarve",
        "split_caps": {"val": 300},
        "max_per_dataset": 2000
    },
    "mgsm": {
        "languages":        None,
        "max_per_language": 50,
        "max_per_dataset":  250,
        "split_caps":       {"val": 100},
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
# Config helpers
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
# COT extraction
# ---------------------------------------------------------------------------

def split_distilled_answer(clean: str, gold: str) -> Tuple[str, str]:
    """Strip trailing \\boxed{} lines → (cot, final_answer)."""
    lines = clean.strip().splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    while lines and "\\boxed{" in lines[-1]:
        lines.pop()
        while lines and not lines[-1].strip():
            lines.pop()
    return "\n".join(lines).strip(), f"\\boxed{{{gold}}}"


def _make_distilled_assistant_message(cot: str, gold: str) -> str:
    answer = gold.strip().upper()
    boxed  = f"\\boxed{{{answer}}}"
    if cot:
        return f"<think>\n{cot}\n</think>\n\n{boxed}"
    return f"<think>\n</think>\n\n{boxed}"

# ---------------------------------------------------------------------------
# SFT record helpers
# ---------------------------------------------------------------------------

def _hash_source_id(ex: Dict[str, Any]) -> str:
    raw = "\n".join([
        str(ex["dataset"]), str(ex["language"]),
        str(ex["task_type"]), str(ex["prompt"]).strip(),
    ])
    return f"{ex['dataset']}_{ex['language']}_{hashlib.sha1(raw.encode()).hexdigest()[:16]}"


def _make_assistant_message(ex: Dict[str, Any]) -> str:
    answer = str(ex["answer"]).strip()
    if ex["task_type"] == "mcq":
        answer = answer.upper()
    boxed = f"\\boxed{{{answer}}}"
    cot = ex.get("cot") or ex.get("rationale") or ex.get("reasoning")
    if cot:
        return f"<think>\n{str(cot).strip()}\n</think>\n\n{boxed}"
    return boxed


def _to_sft_record(ex: Dict[str, Any], split: str, idx: int) -> Dict[str, Any]:
    source_id = str(ex.get("source_id") or ex.get("id") or _hash_source_id(ex))
    return {
        "id":        f"{ex['dataset']}_{split}_{ex['language']}_{idx}",
        "source_id": source_id,
        "loader_id": str(ex.get("id", source_id)),
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
        return [json.loads(l) for l in f if l.strip()]

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


def _build_distilled_sft_record(r: Dict, idx: int, split: str) -> Dict:
    gold      = str(r.get("answer", r.get("distill", {}).get("gold", ""))).strip().upper()
    source_id = str(r.get("source_id", ""))
    dataset   = str(r.get("dataset", "unknown"))
    language  = str(r.get("language", "unknown"))
    prompt    = str(r.get("prompt", ""))
    if not prompt and r.get("messages") and len(r["messages"]) >= 2:
        prompt = str(r["messages"][1]["content"])
    clean     = r.get("distill", {}).get("clean", "")
    cot, _    = split_distilled_answer(clean, gold) if clean else ("", "")
    return {
        "id":        f"distilled_{dataset}_{language}_{idx}",
        "source_id": source_id,
        "loader_id": str(r.get("id", source_id)),
        "dataset":   dataset,
        "language":  language,
        "split":     split,
        "task_type": str(r.get("task_type", "mcq")),
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": prompt},
            {"role": "assistant", "content": _make_distilled_assistant_message(cot, gold)},
        ],
        "answer":   gold,
        "metadata": {
            **r.get("metadata", {}),
            "distilled":           True,
            "used_teacher":        r.get("distill", {}).get("used_teacher"),
            "distill_source_file": r.get("_src_file", ""),
        },
    }


def _load_distilled_dirs(
    distilled_dirs: List[Path],
    val_fraction: float = DISTILLED_VAL_FRACTION,
) -> Tuple[List[Dict], List[Dict], Set[str]]:
    """
    Scan distilled dirs for *_distilled.jsonl files.

    Returns:
        distilled_train: SFT records for train (1 - val_fraction per group)
        distilled_val:   SFT records for val   (val_fraction per group)
        distilled_source_ids: all attempted source_ids (for raw train exclusion)
    """
    raw_records: List[Dict] = []
    distilled_source_ids: Set[str] = set()

    for distilled_dir in distilled_dirs:
        if not distilled_dir.exists():
            print(f"  [WARN] distilled_dir not found: {distilled_dir}")
            continue

        for path in sorted(distilled_dir.glob("*_distilled.jsonl")):
            records = _load_jsonl(path)
            print(f"  Loading distilled: {path.name} ({len(records)} valid)")
            for r in records:
                r["_src_file"] = path.name
                raw_records.append(r)

        # Collect all attempted source_ids from *_all.jsonl
        all_files = [p for p in sorted(distilled_dir.glob("*_all.jsonl"))
                     if "_shuffle_meta" not in p.name]
        if all_files:
            for path in all_files:
                for r in _load_jsonl(path):
                    sid = str(r.get("source_id", ""))
                    if sid:
                        distilled_source_ids.add(sid)
        else:
            for r in raw_records:
                sid = str(r.get("source_id", ""))
                if sid:
                    distilled_source_ids.add(sid)
            print(f"  [INFO] No *_all.jsonl found — using distilled source_ids for exclusion")

    # Deduplicate by prompt + answer
    seen: Set[str] = set()
    deduped: List[Dict] = []
    conflicts: Dict[str, Set[str]] = defaultdict(set)

    for r in raw_records:
        prompt = ""
        if r.get("messages") and len(r["messages"]) >= 2:
            prompt = r["messages"][1]["content"]
        elif r.get("prompt"):
            prompt = r["prompt"]
        prompt_key = hashlib.sha1("\n".join([
            str(r.get("dataset", "")), str(r.get("language", "")),
            str(r.get("task_type", "")), _normalize_for_dedupe(prompt),
        ]).encode("utf-8")).hexdigest()
        conflicts[prompt_key].add(str(r.get("answer", "")).strip().upper())
        key = _distilled_dedupe_key(r)
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    n_conflicts = sum(1 for v in conflicts.values() if len(v) > 1)
    removed     = len(raw_records) - len(deduped)
    if n_conflicts:
        print(f"  [WARN] {n_conflicts} prompts with conflicting answers")
    if removed:
        print(f"  Removed {removed} duplicate prompt+answer records")

    # Split into train/val per dataset+language group with fixed seed
    rng_split = random.Random(VAL_SEED)

    by_group: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for r in deduped:
        by_group[(str(r.get("dataset", "unknown")), str(r.get("language", "unknown")))].append(r)

    distilled_train: List[Dict] = []
    distilled_val:   List[Dict] = []
    idx_train = idx_val = 0

    for (dataset, language), group in sorted(by_group.items()):
        rng_split.shuffle(group)
        n_val = max(1, int(len(group) * val_fraction))
        val_group   = group[:n_val]
        train_group = group[n_val:]

        for r in train_group:
            distilled_train.append(_build_distilled_sft_record(r, idx_train, "train"))
            idx_train += 1
        for r in val_group:
            distilled_val.append(_build_distilled_sft_record(r, idx_val, "val"))
            idx_val += 1

        print(f"  {dataset}/{language}: {len(train_group)} distilled train, "
              f"{len(val_group)} distilled val")

    print(f"  Total distilled: {len(distilled_train)} train, {len(distilled_val)} val")
    print(f"  Attempted source_ids excluded from raw train: {len(distilled_source_ids)}")
    return distilled_train, distilled_val, distilled_source_ids

# ---------------------------------------------------------------------------
# Loader dispatch
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
    if dataset == "milu":
        return load_milu_examples(n=n, split=split, hf_token=hf_token)
    if dataset == "ceval":
        return load_ceval_examples(n=n, split=split)
    if dataset == "cultsportqa":
        return load_cultsportqa_examples(language=language, n=n, split=split)
    if dataset == "belebele":
        return load_belebele_examples(language=language, n=n, split=split)
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
        raise ValueError(f"Recarve not implemented for: {dataset}")
    result = {}
    if val_ex:
        result["val"]  = _dedupe(_to_records(val_ex,  "val"))
    if test_ex:
        result["test"] = _dedupe(_to_records(test_ex, "test"))
    return result

# ---------------------------------------------------------------------------
# Stratified sampling
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
# Mixed file builder
# ---------------------------------------------------------------------------

def _build_mixed(
    per_dir: Path,
    split: str,
    global_max_per_language: int,
    global_max_per_dataset: int,
    max_per_language_mixed: Optional[int],
    rng: random.Random,
    debug: bool,
    extra_records: Optional[List[Dict]] = None,  # distilled train or val to append
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
            # Raw per-dataset file
            path = per_dir / f"{dataset}_{language}_{split}.jsonl"
            if path.exists():
                records = _load_jsonl(path)
                if records:
                    rng.shuffle(records)
                    if not debug:
                        cap = _max_per_language(dataset, language, global_max_per_language)
                        records = (_stratified_sample(records, stratify_by, subject_caps, cap, rng)
                                   if stratify_by else records[:cap])
                    dataset_pool.extend(records)

            # Distilled per-dataset file for this split
            distilled_path = per_dir / f"{dataset}_{language}_distilled_{split}.jsonl"
            if distilled_path.exists():
                dist_records = _load_jsonl(distilled_path)
                if dist_records:
                    rng.shuffle(dist_records)
                    if not debug:
                        cap = _max_per_language(dataset, language, global_max_per_language)
                        dist_records = dist_records[:cap]
                    dataset_pool.extend(dist_records)

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


def _cap_records_by_language(
    records: List[Dict],
    max_per_language: Optional[int],
    rng: random.Random,
    debug: bool,
) -> List[Dict]:
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
    distilled_val_fraction: float = DISTILLED_VAL_FRACTION,
) -> None:
    per_dir   = output_dir / "per_dataset"
    mixed_dir = output_dir / "mixed"
    rng       = random.Random(VAL_SEED)
    summary_rows: List[Dict] = []

    # ── Step 0: load and split distilled examples ─────────────────────────────
    distilled_train: List[Dict] = []
    distilled_val:   List[Dict] = []
    distilled_ids:   Set[str]   = set()

    if distilled_dirs:
        print("\n── Loading distilled examples ──")
        distilled_train, distilled_val, distilled_ids = _load_distilled_dirs(
            distilled_dirs, val_fraction=distilled_val_fraction,
        )

    # ── Step 1: per-dataset/language raw files ────────────────────────────────
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

                # Exclude distilled source_ids from raw train only
                if raw.get("train") and distilled_ids:
                    before = len(raw["train"])
                    raw["train"] = [r for r in raw["train"]
                                    if r["source_id"] not in distilled_ids]
                    excluded = before - len(raw["train"])
                    if excluded:
                        print(f"  Excluded {excluded} train examples covered by distillation")

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

    # ── Step 2: write distilled per-dataset/language train + val files ────────
    for split_name, distilled_records in [("train", distilled_train), ("val", distilled_val)]:
        if not distilled_records:
            continue
        by_group: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
        for r in distilled_records:
            by_group[(r["dataset"], r["language"])].append(r)

        for (dataset, language), records in sorted(by_group.items()):
            records = _reindex(records, split_name)
            out_path = per_dir / f"{dataset}_{language}_distilled_{split_name}.jsonl"
            if _write_jsonl(records, out_path, overwrite):
                summary_rows.append({
                    "dataset": f"{dataset}[distilled]", "language": language,
                    "split": split_name, "n": len(records), "path": str(out_path),
                })

    # ── Step 3: mixed files ───────────────────────────────────────────────────
    produced_splits = set()
    for dataset in DATASET_REGISTRY:
        for split in ALL_SPLITS:
            if _split_enabled(dataset, split):
                produced_splits.add(split)
    if distilled_train:
        produced_splits.add("train")
    if distilled_val:
        produced_splits.add("val")

    for split in ALL_SPLITS:
        if split not in produced_splits:
            continue

        lang_cap = (max_per_language_mixed if split == "train"
                    else DEFAULT_MAX_PER_LANGUAGE_MIXED_VAL if split == "val"
                    else None)

        # _build_mixed now also picks up distilled_{split}.jsonl files from per_dir
        mixed = _build_mixed(
            per_dir=per_dir,
            split=split,
            global_max_per_language=max_per_language,
            global_max_per_dataset=max_per_dataset,
            max_per_language_mixed=lang_cap,
            rng=rng,
            debug=debug,
        )

        # Final language cap for train
        if split == "train" and not debug and max_per_language_mixed is not None:
            before = len(mixed)
            mixed  = _cap_records_by_language(mixed, max_per_language_mixed, rng, debug)
            if len(mixed) < before:
                print(f"  Applied final train language cap: {before} → {len(mixed)}")

        if not mixed:
            continue

        mixed_path = mixed_dir / f"{split}_mixed.jsonl"
        if _write_jsonl(mixed, mixed_path, overwrite):
            summary_rows.append({
                "dataset": "mixed", "language": "all",
                "split": split, "n": len(mixed), "path": str(mixed_path),
            })

        # Save INCLUDE-only val subset for benchmarking
        if split == "val":
            include_val = [r for r in mixed if r["dataset"] == "include"]
            if include_val:
                include_path = mixed_dir / "include_val.jsonl"
                if _write_jsonl(include_val, include_path, overwrite):
                    summary_rows.append({
                        "dataset": "include[val]", "language": "all",
                        "split": "val", "n": len(include_val), "path": str(include_path),
                    })

    # ── Step 4: summary CSV ───────────────────────────────────────────────────
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
    parser.add_argument("--config",    type=Path, default=None)
    parser.add_argument("--distilled_dirs", nargs="*", type=Path, default=None)
    parser.add_argument("--invalid_distilled_policy", type=str,
                        default=DEFAULT_INVALID_DISTILLED_POLICY,
                        choices=["drop", "raw_train", "raw_val_test"])
    parser.add_argument("--distilled_val_fraction", type=float,
                        default=DISTILLED_VAL_FRACTION,
                        help="Fraction of distilled examples held out for val (default 0.20).")
    return parser.parse_args()


def main():
    args = parse_args()

    max_per_language = args.max_per_language
    max_per_dataset  = args.max_per_dataset
    max_mixed        = args.max_per_language_mixed if args.max_per_language_mixed > 0 else None
    output_dir       = DEFAULT_SFT_BASE_DIR
    distilled_dirs   = args.distilled_dirs or []
    invalid_policy   = args.invalid_distilled_policy
    dist_val_frac    = args.distilled_val_fraction

    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        max_per_language = cfg.get("max_per_language", max_per_language)
        max_per_dataset  = cfg.get("max_per_dataset",  max_per_dataset)
        max_mixed        = cfg.get("max_per_language_mixed", max_mixed)
        invalid_policy   = cfg.get("invalid_distilled_policy", invalid_policy)
        dist_val_frac    = cfg.get("distilled_val_fraction", dist_val_frac)
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
        print(f"Distilled dirs:       {[str(d) for d in distilled_dirs]}")
        print(f"Distilled val frac:   {dist_val_frac}")
        print(f"Invalid policy:       {invalid_policy}")
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
        distilled_val_fraction=dist_val_frac,
    )


if __name__ == "__main__":
    main()