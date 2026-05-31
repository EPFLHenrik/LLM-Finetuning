"""multilingual/train/build_distilled_sft_data.py

Builds SFT JSONL files exclusively from distilled examples.
No raw loaders — 100% distilled data pipeline.

Distilled examples are split into train/val per dataset/language group:
  train: 1 - val_fraction
  val:   val_fraction (default 0.15)

Test split is disabled by default. Enable with --enable_test_split.

After building, verifies no benchmark examples (by source_id) appear in
train or val splits.

Output layout:
    <output_dir>/per_dataset/<dataset>_<language>_distilled_train.jsonl
    <output_dir>/per_dataset/<dataset>_<language>_distilled_val.jsonl
    <output_dir>/mixed/train_mixed.jsonl
    <output_dir>/mixed/val_mixed.jsonl
    <output_dir>/mixed/include_val.jsonl
    <output_dir>/build_summary.csv

Usage:
    python -m multilingual.train.build_distilled_sft_data \
        --config configs/sft/v14.json --overwrite
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
DEFAULT_BENCHMARK_PATH = REPO / "multilingual" / "data" / "benchmarks" / "benchmark_all.jsonl"
DEBUG_MAX            = 20

DISTILLED_VAL_FRACTION  = 0.15
DISTILLED_TEST_FRACTION = 0.05

DEFAULT_TRAIN_BLOCK_SHUFFLE = True
DEFAULT_TRAIN_BLOCK_SIZE = 16

# ---------------------------------------------------------------------------
# Dataset registry
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
# Benchmark contamination check
# ---------------------------------------------------------------------------

def _load_benchmark_ids(benchmark_path: Path) -> Set[str]:
    if not benchmark_path.exists():
        print(f"  [WARN] Benchmark file not found: {benchmark_path} — skipping check")
        return set()
    ids = set()
    with open(benchmark_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    r = json.loads(line)
                    sid = r.get("source_id") or r.get("id", "")
                    if sid:
                        ids.add(sid)
                except json.JSONDecodeError:
                    pass
    return ids


def _check_benchmark_contamination(
    output_dir: Path,
    benchmark_ids: Set[str],
) -> None:
    """Check all train/val jsonl files for benchmark source_id contamination."""
    if not benchmark_ids:
        return

    print(f"\n── Benchmark contamination check ({len(benchmark_ids)} benchmark IDs) ──")
    total_contaminated = 0

    for path in sorted(output_dir.rglob("*.jsonl")):
        if "test" in path.name:
            continue
        contaminated = []
        for r in _load_jsonl(path):
            sid = r.get("source_id") or r.get("loader_id", "")
            if sid in benchmark_ids:
                contaminated.append(sid)
        if contaminated:
            print(f"  [CONTAMINATION] {path.name}: {len(contaminated)} benchmark examples found!")
            for sid in contaminated[:5]:
                print(f"    - {sid}")
            if len(contaminated) > 5:
                print(f"    ... and {len(contaminated) - 5} more")
            total_contaminated += len(contaminated)

    if total_contaminated == 0:
        print("  [OK] No benchmark contamination found.")
    else:
        print(f"\n  [ERROR] Total contaminated: {total_contaminated} — review before training!")

# ---------------------------------------------------------------------------
# Distillation loading — train/val split (test optional)
# ---------------------------------------------------------------------------

def _load_distilled_dirs(
    distilled_dirs: List[Path],
    val_fraction:   float = DISTILLED_VAL_FRACTION,
    test_fraction:  float = DISTILLED_TEST_FRACTION,
    enable_test:    bool  = False,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Scan distilled dirs for *_distilled.jsonl files.
    Returns (distilled_train, distilled_val, distilled_test).
    Test split is empty unless enable_test=True.
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

    # Split per dataset+language
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

        if enable_test:
            n_test = max(1, int(len(group) * test_fraction))
        else:
            n_test = 0

        n_val       = max(1, int(len(group) * val_fraction))
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

        test_str = f", {len(test_group)} test" if enable_test else ""
        print(f"  {dataset}/{language}: {len(train_group)} train, "
              f"{len(val_group)} val{test_str}")

    print(f"\n  Total: {len(distilled_train)} train, {len(distilled_val)} val"
          + (f", {len(distilled_test)} test" if enable_test else ""))
    return distilled_train, distilled_val, distilled_test

# ---------------------------------------------------------------------------
# Mixed file builder
# ---------------------------------------------------------------------------

def _block_shuffle_train_mixed(
    records: List[Dict],
    rng: random.Random,
    block_size: int,
) -> List[Dict]:
    """
    Reorder the already-selected train mixed records into small homogeneous blocks.
    This does not change which examples are included. It only changes the final
    order written to mixed/train_mixed.jsonl.
    Blocks are grouped by (dataset, language), shuffled globally, then flattened.
    """
    if block_size <= 1:
        rng.shuffle(records)
        return records

    by_group: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for r in records:
        key = (str(r.get("dataset", "unknown")), str(r.get("language", "unknown")))
        by_group[key].append(r)

    blocks: List[List[Dict]] = []
    for group in by_group.values():
        rng.shuffle(group)
        for i in range(0, len(group), block_size):
            blocks.append(group[i:i + block_size])

    rng.shuffle(blocks)

    ordered: List[Dict] = []
    for block in blocks:
        ordered.extend(block)

    return ordered


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
    enable_test: bool = False,
    benchmark_path: Optional[Path] = None,
    train_block_shuffle: bool = DEFAULT_TRAIN_BLOCK_SHUFFLE,
    train_block_size: int = DEFAULT_TRAIN_BLOCK_SIZE,
) -> None:
    per_dir   = output_dir / "per_dataset"
    mixed_dir = output_dir / "mixed"
    rng       = random.Random(VAL_SEED)
    summary_rows: List[Dict] = []

    active_splits = ["train", "val"] + (["test"] if enable_test else [])

    # ── Step 0: load and split distilled examples ─────────────────────────────
    print("\n── Loading distilled examples ──")
    distilled_train, distilled_val, distilled_test = _load_distilled_dirs(
        distilled_dirs,
        val_fraction=distilled_val_fraction,
        test_fraction=distilled_test_fraction,
        enable_test=enable_test,
    )

    # ── Step 1: write per-dataset/language distilled files ────────────────────
    split_records = [("train", distilled_train), ("val", distilled_val)]
    if enable_test:
        split_records.append(("test", distilled_test))

    for split_name, distilled_records in split_records:
        if not distilled_records:
            continue
        by_group: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
        for r in distilled_records:
            by_group[(r["dataset"], r["language"])].append(r)

        for (dataset, language), records in sorted(by_group.items()):
            if split_name == "train":
                cap = _max_per_language(dataset, language, max_per_language)
                records = records[:cap]
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
    for split in active_splits:
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

        if split == "train" and train_block_shuffle:
            mixed = _block_shuffle_train_mixed(
                records=mixed,
                rng=rng,
                block_size=train_block_size,
            )
            print(f"  Applied train block shuffle: block_size={train_block_size}")

        if not mixed:
            continue

        mixed_path = mixed_dir / f"{split}_mixed.jsonl"
        if _write_jsonl(mixed, mixed_path, overwrite):
            summary_rows.append({
                "dataset": "mixed", "language": "all",
                "split": split, "n": len(mixed), "path": str(mixed_path),
            })

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

    # ── Step 4: benchmark contamination check ─────────────────────────────────
    if benchmark_path is not None:
        benchmark_ids = _load_benchmark_ids(benchmark_path)
        _check_benchmark_contamination(output_dir, benchmark_ids)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_per_language",       type=int, default=DEFAULT_MAX_PER_LANGUAGE)
    parser.add_argument("--max_per_dataset",        type=int, default=DEFAULT_MAX_PER_DATASET)
    parser.add_argument("--max_per_language_mixed", type=int, default=DEFAULT_MAX_PER_LANGUAGE_MIXED)
    parser.add_argument("--debug",            action="store_true")
    parser.add_argument("--overwrite",        action="store_true")
    parser.add_argument("--enable_test_split", action="store_true",
                        help="Also generate a test split (disabled by default)")
    parser.add_argument("--config",    type=Path, default=None)
    parser.add_argument("--distilled_dirs", nargs="*", type=Path, default=None)
    parser.add_argument("--distilled_val_fraction",  type=float, default=DISTILLED_VAL_FRACTION)
    parser.add_argument("--distilled_test_fraction", type=float, default=DISTILLED_TEST_FRACTION)
    parser.add_argument("--benchmark_path", type=Path, default=DEFAULT_BENCHMARK_PATH,
                        help="Path to benchmark_all.jsonl for contamination check")
    parser.add_argument("--no_benchmark_check", action="store_true",
                        help="Skip benchmark contamination check")
    parser.add_argument("--train_block_shuffle", action="store_true",
                    help="Reorder train_mixed.jsonl into dataset/language blocks after sample selection.")
    parser.add_argument("--no_train_block_shuffle", action="store_true",
                        help="Disable train block shuffling.")
    parser.add_argument("--train_block_size", type=int, default=DEFAULT_TRAIN_BLOCK_SIZE,
                        help="Block size for train_mixed.jsonl block shuffle.")
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
    enable_test      = args.enable_test_split
    benchmark_path   = None if args.no_benchmark_check else args.benchmark_path
    train_block_shuffle = DEFAULT_TRAIN_BLOCK_SHUFFLE
    if args.train_block_shuffle:
        train_block_shuffle = True
    if args.no_train_block_shuffle:
        train_block_shuffle = False
    train_block_size = args.train_block_size

    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        max_per_language = cfg.get("max_per_language", max_per_language)
        max_per_dataset  = cfg.get("max_per_dataset",  max_per_dataset)
        max_mixed        = cfg.get("max_per_language_mixed", max_mixed)
        dist_val_frac    = cfg.get("distilled_val_fraction",  dist_val_frac)
        dist_test_frac   = cfg.get("distilled_test_fraction", dist_test_frac)
        enable_test      = cfg.get("enable_test_split", enable_test)
        train_block_shuffle = cfg.get("train_block_shuffle", train_block_shuffle)
        train_block_size    = cfg.get("train_block_size", train_block_size)
        if output_name := cfg.get("output_name"):
            output_dir = DEFAULT_SFT_BASE_DIR / output_name
        if "dataset_registry" in cfg:
            DATASET_REGISTRY.update(cfg["dataset_registry"])
        if "distilled_dirs" in cfg:
            distilled_dirs = [Path(d) for d in cfg["distilled_dirs"]]

    print(f"Debug:        {args.debug}  Overwrite: {args.overwrite}")
    print(f"Config:       {args.config or 'default'}")
    print(f"Output dir:   {output_dir}")
    print(f"Test split:   {'enabled' if enable_test else 'disabled'}")
    print(f"Benchmark check: {'disabled' if benchmark_path is None else str(benchmark_path)}")
    print(f"Caps:         max_per_language={max_per_language}, "
          f"max_per_dataset={max_per_dataset}, "
          f"max_per_language_mixed={max_mixed}")
    print(f"Distilled:    val_frac={dist_val_frac}, test_frac={dist_test_frac}")
    print(f"Train ordering: block_shuffle={train_block_shuffle}, block_size={train_block_size}")
    print(f"Dirs:         {[str(d) for d in distilled_dirs]}")
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
        enable_test=enable_test,
        benchmark_path=benchmark_path,
        train_block_shuffle=train_block_shuffle,
        train_block_size=train_block_size,
    )


if __name__ == "__main__":
    main()