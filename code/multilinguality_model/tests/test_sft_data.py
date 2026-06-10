"""Tests for multilinguality SFT data in the M3 repo.

Expected M3 layout:

    data/multilinguality/sft/<version>/
        train_mixed.jsonl
        val_mixed.jsonl
        build_summary.csv

Run all available SFT versions:

    python -m pip install pytest
    python -m pytest code/multilinguality_model/tests/test_sft_data.py -v

Run selected versions only:

    SFT_VERSIONS="v1,v20-russian,v20-chinese,v20-hindi,v20-italian,v20-spanish" \
    python -m pytest code/multilinguality_model/tests/test_sft_data.py -v
"""

import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths: M3 repo layout
# ---------------------------------------------------------------------------

REPO_DIR = Path(__file__).resolve().parents[3]
CODE_DIR = REPO_DIR / "code"
SFT_ROOT = REPO_DIR / "data" / "multilinguality" / "sft"

sys.path.insert(0, str(CODE_DIR))

from multilinguality_model.eval.extract_answer import (  # noqa: E402
    extract_boxed_answer,
    last_boxed_only_string,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = {
    "id",
    "source_id",
    "loader_id",
    "dataset",
    "language",
    "split",
    "task_type",
    "messages",
    "answer",
    "metadata",
}

VALID_SPLITS = {"train", "val", "test"}
VALID_TASK_TYPES = {"mcq", "math"}
VALID_ROLES = ["system", "user", "assistant"]
MCQ_ANSWER_RE = re.compile(r"^[A-Z]$")


# ---------------------------------------------------------------------------
# SFT version discovery
# ---------------------------------------------------------------------------

def _sft_versions():
    requested = os.environ.get("SFT_VERSIONS")

    if requested:
        versions = [v.strip() for v in requested.split(",") if v.strip()]
        return [SFT_ROOT / version for version in versions]

    if not SFT_ROOT.exists():
        return []

    return sorted(path for path in SFT_ROOT.iterdir() if path.is_dir())


SFT_DATA_DIRS = _sft_versions()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _line_count(path: Path):
    return sum(1 for line in open(path, encoding="utf-8") if line.strip())


def _flat_mixed_file(sft_dir: Path, split: str) -> Path:
    return sft_dir / f"{split}_mixed.jsonl"


def _legacy_mixed_file(sft_dir: Path, split: str) -> Path:
    return sft_dir / "mixed" / f"{split}_mixed.jsonl"


def _mixed_file_for_split(sft_dir: Path, split: str):
    flat = _flat_mixed_file(sft_dir, split)
    legacy = _legacy_mixed_file(sft_dir, split)

    if flat.exists():
        return flat

    if legacy.exists():
        return legacy

    return flat


def _all_mixed_files():
    files = []

    for sft_dir in SFT_DATA_DIRS:
        for split in ("train", "val", "test"):
            path = _mixed_file_for_split(sft_dir, split)
            if path.exists():
                files.append(path)

    return sorted(files)


def _all_per_dataset_files():
    files = []

    for sft_dir in SFT_DATA_DIRS:
        per_dir = sft_dir / "per_dataset"
        if per_dir.exists():
            files.extend(sorted(per_dir.glob("*.jsonl")))

    return files


def _summary_rows():
    rows = []

    for sft_dir in SFT_DATA_DIRS:
        path = sft_dir / "build_summary.csv"
        if not path.exists():
            continue

        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["_sft_dir"] = str(sft_dir)
                rows.append(row)

    return rows


def _resolve_summary_path(row):
    """Resolve build_summary.csv paths robustly.

    Some summary files may contain stale absolute M2 paths. For M3 tests, we
    always prefer files inside the current SFT version directory.
    """
    sft_dir = Path(row["_sft_dir"])
    raw = Path(row.get("path", ""))

    filename = raw.name

    if filename in {"train_mixed.jsonl", "val_mixed.jsonl", "test_mixed.jsonl"}:
        candidate = sft_dir / filename
        if candidate.exists():
            return candidate

        candidate = sft_dir / "mixed" / filename
        if candidate.exists():
            return candidate

    if not raw.is_absolute():
        candidate = sft_dir / raw
        if candidate.exists():
            return candidate

        candidate = REPO_DIR / raw
        if candidate.exists():
            return candidate

    if raw.is_absolute() and str(raw).startswith(str(REPO_DIR)) and raw.exists():
        return raw

    return None


def _split_from_mixed_filename(path: Path):
    return path.name.replace("_mixed.jsonl", "")


def _split_from_per_dataset_filename(path: Path):
    return path.stem.rsplit("_", 1)[-1]


def _normalize_language(language):
    return str(language).strip().lower().replace("_", "-").replace(" ", "-")


def _check_schema(records, path):
    assert len(records) > 0, f"Empty file: {path}"

    for r in records:
        missing = REQUIRED_FIELDS - set(r.keys())
        assert not missing, f"Missing fields {missing} in {r.get('id', '?')} ({path})"

        assert r["split"] in VALID_SPLITS, f"Bad split '{r['split']}' in {r['id']}"
        assert r["task_type"] in VALID_TASK_TYPES, f"Bad task_type in {r['id']}"

        assert isinstance(r["messages"], list), f"messages not a list in {r['id']}"
        assert isinstance(r["metadata"], dict), f"metadata not a dict in {r['id']}"


def _check_loader_id(records):
    for r in records:
        assert r.get("loader_id"), f"loader_id missing or empty in {r['id']}"
        assert r.get("source_id"), f"source_id missing or empty in {r['id']}"
        assert isinstance(r["source_id"], str) and len(r["source_id"]) > 0


def _check_messages(records):
    for r in records:
        msgs = r["messages"]

        assert len(msgs) == 3, f"Expected 3 messages in {r['id']}, got {len(msgs)}"

        for msg, role in zip(msgs, VALID_ROLES):
            assert msg["role"] == role, (
                f"Role mismatch in {r['id']}: expected {role}, got {msg['role']}"
            )
            assert isinstance(msg["content"], str) and msg["content"].strip(), (
                f"Empty content in {r['id']} role={msg['role']}"
            )


def _check_assistant_format(records):
    for r in records:
        assistant = r["messages"][2]["content"].strip()
        last_line = assistant.split("\n")[-1].strip()

        if "<think>" in assistant or "</think>" in assistant:
            assert "</think>" in assistant, f"Has <think> but no </think> in {r['id']}"

            after_think = assistant.split("</think>")[-1].strip()
            assert "\\boxed{" in after_think, f"No \\boxed{{}} after </think> in {r['id']}"

            boxed_expr = last_boxed_only_string(after_think)
            assert boxed_expr is not None, f"No boxed expression after </think> in {r['id']}"

            assert last_line == boxed_expr, (
                f"Last line is not boxed answer in {r['id']}: '{last_line}'"
            )

        else:
            all_boxed = re.findall(r"\\boxed\{", assistant)
            assert len(all_boxed) == 1, (
                f"Expected 1 \\boxed{{}}, found {len(all_boxed)} in {r['id']}"
            )

            boxed_expr = last_boxed_only_string(assistant)
            assert boxed_expr is not None, f"No boxed expression in {r['id']}"

            assert assistant == boxed_expr, (
                f"Raw assistant should be exactly boxed answer in {r['id']}: '{assistant}'"
            )


def _check_answer_normalization(records):
    for r in records:
        assistant = r["messages"][2]["content"]
        boxed = extract_boxed_answer(assistant)

        assert boxed is not None, f"No boxed answer in {r['id']}"

        if r["task_type"] == "mcq":
            assert MCQ_ANSWER_RE.match(str(r["answer"])), (
                f"Bad MCQ answer field '{r['answer']}' in {r['id']}"
            )
            assert MCQ_ANSWER_RE.match(boxed.strip()), (
                f"Boxed MCQ answer '{boxed}' is not A-Z in {r['id']}"
            )
            assert boxed.strip() == str(r["answer"]).strip(), (
                f"Boxed '{boxed}' != answer '{r['answer']}' in {r['id']}"
            )

        else:
            assert str(r["answer"]).strip(), f"Empty math answer in {r['id']}"
            assert boxed.strip() == str(r["answer"]).strip(), (
                f"Boxed '{boxed}' != answer '{r['answer']}' in {r['id']}"
            )


# ---------------------------------------------------------------------------
# Basic existence
# ---------------------------------------------------------------------------

def test_sft_root_exists():
    assert SFT_ROOT.exists(), f"Missing SFT root: {SFT_ROOT}"


def test_sft_versions_exist():
    assert len(SFT_DATA_DIRS) > 0, f"No SFT version folders found under {SFT_ROOT}"

    missing = [str(path) for path in SFT_DATA_DIRS if not path.exists()]
    assert not missing, f"Missing requested SFT dirs: {missing}"


@pytest.mark.parametrize("sft_dir", SFT_DATA_DIRS)
def test_sft_version_has_expected_structure(sft_dir):
    assert sft_dir.exists(), f"Missing SFT dir: {sft_dir}"
    assert (sft_dir / "build_summary.csv").exists(), f"Missing build_summary.csv in {sft_dir}"

    has_flat = (
        (sft_dir / "train_mixed.jsonl").exists()
        and (sft_dir / "val_mixed.jsonl").exists()
    )
    has_legacy = (
        (sft_dir / "mixed" / "train_mixed.jsonl").exists()
        and (sft_dir / "mixed" / "val_mixed.jsonl").exists()
    )

    assert has_flat or has_legacy, (
        f"Expected flat train_mixed/val_mixed or legacy mixed/ files in {sft_dir}"
    )


def test_mixed_files_exist():
    missing = []

    for sft_dir in SFT_DATA_DIRS:
        for split in ("train", "val"):
            path = _mixed_file_for_split(sft_dir, split)
            if not path.exists():
                missing.append(str(path))

    assert not missing, f"Missing required mixed files: {missing}"


def test_summary_csv_exists():
    missing = [
        str(sft_dir / "build_summary.csv")
        for sft_dir in SFT_DATA_DIRS
        if not (sft_dir / "build_summary.csv").exists()
    ]
    assert not missing, f"Missing summary CSV files: {missing}"


# ---------------------------------------------------------------------------
# Mixed file checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", _all_mixed_files())
def test_mixed_schema(path):
    _check_schema(_load_jsonl(path), path)


@pytest.mark.parametrize("path", _all_mixed_files())
def test_mixed_loader_ids(path):
    _check_loader_id(_load_jsonl(path))


@pytest.mark.parametrize("path", _all_mixed_files())
def test_mixed_messages(path):
    _check_messages(_load_jsonl(path))


@pytest.mark.parametrize("path", _all_mixed_files())
def test_mixed_assistant_format(path):
    _check_assistant_format(_load_jsonl(path))


@pytest.mark.parametrize("path", _all_mixed_files())
def test_mixed_answer_normalization(path):
    _check_answer_normalization(_load_jsonl(path))


@pytest.mark.parametrize("path", _all_mixed_files())
def test_mixed_split_label_consistent(path):
    expected = _split_from_mixed_filename(path)

    for r in _load_jsonl(path):
        assert r["split"] == expected, (
            f"Row {r['id']} has split='{r['split']}' in {path}"
        )


@pytest.mark.parametrize("path", _all_mixed_files())
def test_mixed_unique_ids(path):
    ids = [r["id"] for r in _load_jsonl(path)]
    assert len(ids) == len(set(ids)), f"Duplicate IDs in {path}"


@pytest.mark.parametrize("path", _all_mixed_files())
def test_mixed_no_unexpected_split_labels(path):
    bad = {r["split"] for r in _load_jsonl(path)} - VALID_SPLITS
    assert not bad, f"Unexpected split labels in {path}: {bad}"


# ---------------------------------------------------------------------------
# Optional per-dataset checks, only if legacy per_dataset exists
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", _all_per_dataset_files())
def test_per_dataset_schema(path):
    _check_schema(_load_jsonl(path), path)


@pytest.mark.parametrize("path", _all_per_dataset_files())
def test_per_dataset_loader_ids(path):
    _check_loader_id(_load_jsonl(path))


@pytest.mark.parametrize("path", _all_per_dataset_files())
def test_per_dataset_messages(path):
    _check_messages(_load_jsonl(path))


@pytest.mark.parametrize("path", _all_per_dataset_files())
def test_per_dataset_assistant_format(path):
    _check_assistant_format(_load_jsonl(path))


@pytest.mark.parametrize("path", _all_per_dataset_files())
def test_per_dataset_answer_normalization(path):
    _check_answer_normalization(_load_jsonl(path))


@pytest.mark.parametrize("path", _all_per_dataset_files())
def test_per_dataset_split_label_matches_filename(path):
    expected = _split_from_per_dataset_filename(path)

    for r in _load_jsonl(path):
        assert r["split"] == expected, (
            f"split '{r['split']}' != filename split '{expected}' in {r['id']} ({path})"
        )


@pytest.mark.parametrize("path", _all_per_dataset_files())
def test_per_dataset_unique_ids(path):
    ids = [r["id"] for r in _load_jsonl(path)]
    assert len(ids) == len(set(ids)), f"Duplicate IDs in {path}"


@pytest.mark.parametrize("path", _all_per_dataset_files())
def test_per_dataset_unique_source_ids(path):
    source_ids = [r["source_id"] for r in _load_jsonl(path)]
    dupes = len(source_ids) - len(set(source_ids))
    assert dupes == 0, f"{dupes} duplicate source_ids in {path}"


@pytest.mark.parametrize("sft_dir", SFT_DATA_DIRS)
def test_no_train_val_overlap_per_dataset_if_present(sft_dir):
    per_dir = sft_dir / "per_dataset"

    if not per_dir.exists():
        pytest.skip(f"No per_dataset dir in {sft_dir}")

    by_key = defaultdict(lambda: {"train": set(), "val": set()})

    for path in sorted(per_dir.glob("*.jsonl")):
        split = _split_from_per_dataset_filename(path)

        if split not in {"train", "val"}:
            continue

        for r in _load_jsonl(path):
            by_key[(r["dataset"], r["language"])][split].add(r["source_id"])

    for (dataset, language), splits in by_key.items():
        overlap = splits["train"] & splits["val"]
        assert not overlap, (
            f"{len(overlap)} overlapping source_ids in {dataset}/{language} for {sft_dir.name}"
        )


# ---------------------------------------------------------------------------
# Train/val overlap checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sft_dir", SFT_DATA_DIRS)
def test_mixed_no_train_val_overlap(sft_dir):
    train_path = _mixed_file_for_split(sft_dir, "train")
    val_path = _mixed_file_for_split(sft_dir, "val")

    if not train_path.exists() or not val_path.exists():
        pytest.skip(f"Missing train or val mixed file in {sft_dir}")

    train_keys = {
        (r["dataset"], r["language"], r["source_id"])
        for r in _load_jsonl(train_path)
    }
    val_keys = {
        (r["dataset"], r["language"], r["source_id"])
        for r in _load_jsonl(val_path)
    }

    overlap = train_keys & val_keys
    assert not overlap, (
        f"{len(overlap)} examples in both mixed train and val for {sft_dir.name}"
    )


# ---------------------------------------------------------------------------
# Summary CSV consistency
# ---------------------------------------------------------------------------

def test_summary_counts_match_files():
    checked = 0

    for row in _summary_rows():
        path = _resolve_summary_path(row)

        if path is None or not path.exists():
            continue

        actual = _line_count(path)
        expected = int(row["n"])

        assert actual == expected, (
            f"Summary says {expected} but {path} has {actual} lines"
        )

        checked += 1

    assert checked > 0, "No build_summary.csv file paths could be resolved to M3 files"


@pytest.mark.parametrize("path", _all_mixed_files())
def test_mixed_counts_match_summary(path):
    sft_dir = path.parent
    split = _split_from_mixed_filename(path)

    if path.parent.name == "mixed":
        sft_dir = path.parents[1]

    summary = {
        (row["dataset"], row["split"]): int(row["n"])
        for row in _summary_rows()
        if row.get("_sft_dir") == str(sft_dir) and row["dataset"] == "mixed"
    }

    actual = _line_count(path)
    assert actual > 0, f"{path} is empty"

    if ("mixed", split) in summary:
        expected = summary[("mixed", split)]
        assert actual == expected, (
            f"Summary says {expected} but {path} has {actual} lines"
        )


# ---------------------------------------------------------------------------
# Distribution checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", _all_mixed_files())
def test_mixed_multiple_datasets(path):
    split = _split_from_mixed_filename(path)
    datasets = {r["dataset"] for r in _load_jsonl(path)}

    min_datasets = 2 if split == "test" else 1

    assert len(datasets) >= min_datasets, (
        f"{path} only has datasets: {datasets}"
    )


@pytest.mark.parametrize("path", _all_mixed_files())
def test_mixed_multiple_languages(path):
    languages = {r["language"] for r in _load_jsonl(path)}

    assert len(languages) >= 1, (
        f"{path} has no valid languages: {languages}"
    )


@pytest.mark.parametrize("sft_dir", SFT_DATA_DIRS)
def test_monolingual_expert_language_consistency(sft_dir):
    version = sft_dir.name

    expected_by_version = {
        "v20-russian": {"russian", "ru"},
        "v20-chinese": {"chinese", "zh", "zh-cn", "zh-hans"},
        "v20-hindi": {"hindi", "hi"},
        "v20-italian": {"italian", "it"},
        "v20-spanish": {"spanish", "es"},
    }

    if version not in expected_by_version:
        pytest.skip(f"{version} is not a monolingual expert version")

    expected = expected_by_version[version]

    mixed_files = [
        path
        for split in ("train", "val", "test")
        if (path := _mixed_file_for_split(sft_dir, split)).exists()
    ]

    assert mixed_files, f"No mixed files found for {version}"

    for path in mixed_files:
        records = _load_jsonl(path)
        languages = {_normalize_language(r["language"]) for r in records}

        assert languages <= expected, (
            f"{version}/{path.name} expected languages in {expected}, got {languages}"
        )