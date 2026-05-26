"""multilingual/train/build_distilled_sft_data.py

Builds SFT JSONL files exclusively from distilled examples.
No raw loaders — 100% distilled data pipeline.

Distilled examples are split into train/val/test per dataset/language group:
  train: 1 - val_fraction - test_fraction
  val:   val_fraction  (default 0.15)
  test:  test_fraction (default 0.05)

Output layout:
    <output_dir>/per_dataset/<dataset>_<language>_distilled_train.jsonl
    <output_dir>/per_dataset/<dataset>_<language>_distilled_val.jsonl
    <output_dir>/per_dataset/<dataset>_<language>_distilled_test.jsonl
    <output_dir>/mixed/train_mixed.jsonl
    <output_dir>/mixed/val_mixed.jsonl
    <output_dir>/mixed/test_mixed.jsonl
    <output_dir>/mixed/include_val.jsonl
    <output_dir>/build_summary.csv

Usage:
    python -m multilingual.train.build_distilled_sft_data \
        --config configs/sft/v11.json --overwrite
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

from multilingual.config import REPO, SYSTEM_PROMPT, TARGET_LANGUAGES, VAL_SEED

# ---------------------------------------------------------------------------
# Global defaults
# ---------------------------------------------------------------------------

DEFAULT_MAX_PER_LANGUAGE           = 5000
DEFAULT_MAX_PER_DATASET            = 10000
DEFAULT_MAX_PER_LANGUAGE_MIXED     = 2500
DEFAULT_MAX_PER_LANGUAGE_MIXED_VAL = 1000
DEFAULT_SFT_BASE_DIR = REPO / "multilingual" / "data" / "sft"
DEBUG_MAX            = 20
ALL_SPLITS           = ("train", "val", "test")

DISTILLED_VAL_FRACTION  = 0.15
DISTILLED_TEST_FRACTION = 0.05

# ---------------------------------------------------------------------------
# Dataset registry — caps reflect distillation pool sizes
# ---------------------------------------------------------------------------

DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "include": {
        "languages": ["italian", "spanish", "russian", "chinese", "hindi"],
        "split_caps": {"val": 500, "test": 100},
    },
    "global_piqa": {
        "languages": None,
        "split_caps": {"val": 200, "test": 50},
        "max_per_dataset": 2000,
    },
    "mgsm": {
        "languages": None,
        "max_per_language": 100,
        "max_per_dataset": 500,
        "split_caps": {"val": 100},
    },
    "milu": {
        "languages": ["hindi"],
        "language_caps": {"hindi": 5000},
        "max_per_dataset": 5000,
        "split_caps": {"val": 400, "test": 100},
    },
    "ceval": {
        "languages": ["chinese"],
        "language_caps": {"chinese": 5000},
        "max_per_dataset": 5000,
        "split_caps": {"val": 400, "test": 100},
    },
    "cultsportqa": {
        "languages": ["chinese", "hindi"],
        "max_per_language": 1500,
        "max_per_dataset": 3000,
        "split_caps": {"val": 300, "test": 50},
    },
    "exams": {
        "languages": ["italian", "spanish"],
        "split_caps": {"val": 200, "test": 50},
    },
    "head_qa": {
        "languages": ["spanish", "italian"],
        "language_caps": {"spanish": 2000, "italian": 1500},
        "max_per_dataset": 3500,
        "split_caps": {"val": 300, "test": 50},
    },
    "mera": {
        "languages": ["russian"],
        "max_per_language": 3000,
        "max_per_dataset": 3000,
        "split_caps": {"val": 200, "test": 50},
    },
    "global_mmlu": {
        "languages": ["russian", "chinese", "hindi", "italian", "spanish"],
        "language_caps": {
            "russian": 2000,
            "chinese": 800,
            "hindi":   800,
            "italian": 2000,
            "spanish": 2000,
        },
        "max_per_dataset": 5000,
        "split_caps": {"val": 600, "test": 100},
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

# ---------------------------------------------------------------------------
# COT extraction
# ---------------------------------------------------------------------------

def split_distilled_answer(clean: str, gold: str) -> Tuple[str, str]:
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

def _normalize_for_dedupe(x: str) -> str:
    return re.sub(r"\s+", " ", str(x).strip())


def _distilled_dedupe_key(r: Dict[str, Any]) -> str:
    prompt = ""
    if r.get("messages") and len(r["messages"]) >= 2:
        prompt = r["messages"][1]["content"]
    elif r.get("prompt"):
        prompt = r["prompt"]
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
    clean  = r.get("distill", {}).get("clean", "")
    cot, _ = split_distilled_answer(clean, gold) if clean else ("", "")
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
# Distillation loading — 3-way train/val/test split
# ---------------------------------------------------------------------------

def _load_distilled_dirs(
    distilled_dirs: List[Path],
    val_fraction:  float = DISTILLED_VAL_FRACTION,
    test_fraction: float = DISTILLED_TEST_FRACTION,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Scan distilled dirs for *_distilled.jsonl files.
    Returns (distilled_train, distilled_val, distilled_test).
    Split is per dataset+language group with fixed seed.
    """
    raw_records: List[Dict] = []

    for distilled_dir in distilled_dirs:
        if not distilled_dir.exists():
            print(f"  [WARN] distilled_dir not found: {distilled_dir}")
            continue
        for path in sorted(distilled_dir.glob("*_distilled.jsonl")):
            records = _load_jsonl(path)
            print(f"  Loading distilled: {path.name} ({len(records)} examples)")
            for r in records:
                r["_src_file"] = path.name
                raw_records.append(r)

    # Deduplicate
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
        print(f"  Removed {removed} duplicate records")

    # 3-way split per dataset+language
    rng = random.Random(VAL_SEED)

    by_group: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for r in deduped:
        by_group[(str(r.get("dataset", "unknown")), str(r.get("language", "unknown")))].append(r)

    distilled_train: List[Dict] = []
    distilled_val:   List[Dict] = []
    distilled_test:  List[Dict] = []
    idx_train = idx_val = idx_test = 0

    for (dataset, language), group in sorted(by_group.items()):
        rng.shuffle(group)
        n_test = max(1, int(len(group) * test_fraction))
        n_val  = max(1, int(len(group) * val_fraction))
        test_group  = group[:n_test]
        val_group   = group[n_test:n_test + n_val]
        train_group = group[n_test + n_val:]

        for r in train_group:
            distilled_train.append(_build_distilled_sft_record(r, idx_train, "train"))
            idx_train += 1
        for r in val_group:
            distilled_val.append(_build_distilled_sft_record(r, idx_val, "val"))
            idx_val += 1
        for r in test_group:
            distilled_test.append(_build_distilled_sft_record(r, idx_test, "test"))
            idx_test += 1

        print(f"  {dataset}/{language}: {len(train_group)} train, "
              f"{len(val_group)} val, {len(test_group)} test")

    print(f"\n  Total: {len(distilled_train)} train, "
          f"{len(distilled_val)} val, {len(distilled_test)} test")
    return distilled_train, distilled_val, distilled_test

# ---------------------------------------------------------------------------
# Mixed file builder — distilled only
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
        dataset_pool: List[Dict] = []

        for language in _languages(dataset):
            distilled_path = per_dir / f"{dataset}_{language}_distilled_{split}.jsonl"
            if not distilled_path.exists():
                continue
            records = _load_jsonl(distilled_path)
            if not records:
                continue
            rng.shuffle(records)
            if not debug:
                cap = _max_per_language(dataset, language, global_max_per_language)
                records = records[:cap]
            dataset_pool.extend(records)

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
    distilled_dirs: List[Path],
    distilled_val_fraction:  float = DISTILLED_VAL_FRACTION,
    distilled_test_fraction: float = DISTILLED_TEST_FRACTION,
) -> None:
    per_dir   = output_dir / "per_dataset"
    mixed_dir = output_dir / "mixed"
    rng       = random.Random(VAL_SEED)
    summary_rows: List[Dict] = []

    # ── Step 0: load and split distilled examples ─────────────────────────────
    print("\n── Loading distilled examples ──")
    distilled_train, distilled_val, distilled_test = _load_distilled_dirs(
        distilled_dirs,
        val_fraction=distilled_val_fraction,
        test_fraction=distilled_test_fraction,
    )

    # ── Step 1: write per-dataset/language distilled files ────────────────────
    for split_name, distilled_records in [
        ("train", distilled_train),
        ("val",   distilled_val),
        ("test",  distilled_test),
    ]:
        if not distilled_records:
            continue
        by_group: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
        for r in distilled_records:
            by_group[(r["dataset"], r["language"])].append(r)

        for (dataset, language), records in sorted(by_group.items()):
            # Apply per-language cap to train
            if split_name == "train":
                cap = _max_per_language(dataset, language, max_per_language)
                records = records[:cap]
            # Apply split caps to val/test
            elif split_name in ("val", "test"):
                cap = _split_cap(dataset, split_name, max_per_dataset)
                if cap < 10_000_000:
                    records = records[:cap]

            records  = _reindex(records, split_name)
            out_path = per_dir / f"{dataset}_{language}_distilled_{split_name}.jsonl"
            if _write_jsonl(records, out_path, overwrite):
                summary_rows.append({
                    "dataset": f"{dataset}[distilled]", "language": language,
                    "split": split_name, "n": len(records), "path": str(out_path),
                })

    # ── Step 2: mixed files ───────────────────────────────────────────────────
    for split in ALL_SPLITS:
        lang_cap = (max_per_language_mixed      if split == "train"
                    else DEFAULT_MAX_PER_LANGUAGE_MIXED_VAL if split == "val"
                    else None)

        mixed = _build_mixed(
            per_dir=per_dir,
            split=split,
            global_max_per_language=max_per_language,
            global_max_per_dataset=max_per_dataset,
            max_per_language_mixed=lang_cap,
            rng=rng,
            debug=debug,
        )

        if not mixed:
            continue

        mixed_path = mixed_dir / f"{split}_mixed.jsonl"
        if _write_jsonl(mixed, mixed_path, overwrite):
            summary_rows.append({
                "dataset": "mixed", "language": "all",
                "split": split, "n": len(mixed), "path": str(mixed_path),
            })

        # INCLUDE-only val subset for benchmarking
        if split == "val":
            include_val = [r for r in mixed if r["dataset"] == "include"]
            if include_val:
                include_path = mixed_dir / "include_val.jsonl"
                if _write_jsonl(include_val, include_path, overwrite):
                    summary_rows.append({
                        "dataset": "include[val]", "language": "all",
                        "split": "val", "n": len(include_val), "path": str(include_path),
                    })

    # ── Step 3: summary CSV ───────────────────────────────────────────────────
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
    parser.add_argument("--distilled_val_fraction",  type=float, default=DISTILLED_VAL_FRACTION)
    parser.add_argument("--distilled_test_fraction", type=float, default=DISTILLED_TEST_FRACTION)
    return parser.parse_args()


def main():
    args = parse_args()

    max_per_language = args.max_per_language
    max_per_dataset  = args.max_per_dataset
    max_mixed        = args.max_per_language_mixed if args.max_per_language_mixed > 0 else None
    output_dir       = DEFAULT_SFT_BASE_DIR
    distilled_dirs   = args.distilled_dirs or []
    dist_val_frac    = args.distilled_val_fraction
    dist_test_frac   = args.distilled_test_fraction

    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        max_per_language = cfg.get("max_per_language", max_per_language)
        max_per_dataset  = cfg.get("max_per_dataset",  max_per_dataset)
        max_mixed        = cfg.get("max_per_language_mixed", max_mixed)
        dist_val_frac    = cfg.get("distilled_val_fraction",  dist_val_frac)
        dist_test_frac   = cfg.get("distilled_test_fraction", dist_test_frac)
        if output_name := cfg.get("output_name"):
            output_dir = DEFAULT_SFT_BASE_DIR / output_name
        if "dataset_registry" in cfg:
            DATASET_REGISTRY.update(cfg["dataset_registry"])
        if "distilled_dirs" in cfg:
            distilled_dirs = [Path(d) for d in cfg["distilled_dirs"]]

    hf_token = os.environ.get("HF_TOKEN", "")  # not used but kept for consistency
    print(f"Debug:      {args.debug}  Overwrite: {args.overwrite}")
    print(f"Config:     {args.config or 'default'}")
    print(f"Output dir: {output_dir}")
    print(f"Caps:       max_per_language={max_per_language}, "
          f"max_per_dataset={max_per_dataset}, "
          f"max_per_language_mixed={max_mixed}")
    print(f"Distilled:  val_frac={dist_val_frac}, test_frac={dist_test_frac}")
    print(f"Dirs:       {[str(d) for d in distilled_dirs]}")
    print()

    build(
        output_dir=output_dir,
        max_per_language=max_per_language,
        max_per_dataset=max_per_dataset,
        max_per_language_mixed=max_mixed,
        debug=args.debug,
        overwrite=args.overwrite,
        distilled_dirs=distilled_dirs or [],
        distilled_val_fraction=dist_val_frac,
        distilled_test_fraction=dist_test_frac,
    )


if __name__ == "__main__":
    main()