"""multilingual/tests/test_sft_data.py

Run with:
    pytest multilingual/tests/test_sft_data.py -v
"""

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MULTILINGUAL_DIR = Path(__file__).parents[1]
SFT_DATA_DIR     = MULTILINGUAL_DIR / "data" / "sft"
REPO_DIR         = MULTILINGUAL_DIR.parent

sys.path.insert(0, str(REPO_DIR))
from evaluate.extract_answer import extract_boxed_answer, last_boxed_only_string

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_FIELDS  = {
    "id", "source_id", "loader_id", "dataset", "language", "split",
    "task_type", "messages", "answer", "metadata",
}
VALID_SPLITS      = {"train", "val"}
VALID_TASK_TYPES  = {"mcq", "math"}
VALID_ROLES       = ["system", "user", "assistant"]
MCQ_ANSWER_RE     = re.compile(r"^[A-D]$")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _all_per_dataset_files():
    per_dir = SFT_DATA_DIR / "per_dataset"
    if not per_dir.exists():
        return []
    return sorted(per_dir.glob("*.jsonl"))


def _mixed_file(split: str) -> Path:
    return SFT_DATA_DIR / "mixed" / f"{split}_mixed.jsonl"


def _summary_rows():
    path = SFT_DATA_DIR / "build_summary.csv"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _ends_with_boxed(text: str) -> bool:
    """Check text ends exactly with its last \\boxed{...} expression."""
    boxed_expr = last_boxed_only_string(text)
    if boxed_expr is None:
        return False
    return text.strip().endswith(boxed_expr)


def _check_schema(records, path):
    assert len(records) > 0, f"Empty file: {path}"
    for r in records:
        missing = REQUIRED_FIELDS - set(r.keys())
        assert not missing, f"Missing fields {missing} in {r.get('id','?')} ({path})"
        assert r["split"] in VALID_SPLITS, f"Bad split '{r['split']}' in {r['id']}"
        assert r["task_type"] in VALID_TASK_TYPES, f"Bad task_type in {r['id']}"
        assert isinstance(r["messages"], list), f"messages not a list in {r['id']}"
        assert isinstance(r["metadata"], dict), f"metadata not a dict in {r['id']}"


def _check_loader_id(records):
    for r in records:
        assert r.get("loader_id"), f"loader_id missing or empty in {r['id']}"
        assert r.get("source_id"), f"source_id missing or empty in {r['id']}"
        # source_id is hash-based so we only check it's a non-empty string
        assert isinstance(r["source_id"], str) and len(r["source_id"]) > 0


def _check_messages(records):
    for r in records:
        msgs = r["messages"]
        assert len(msgs) == 3, f"Expected 3 messages in {r['id']}, got {len(msgs)}"
        for msg, role in zip(msgs, VALID_ROLES):
            assert msg["role"] == role, \
                f"Role mismatch in {r['id']}: expected {role}, got {msg['role']}"
            assert isinstance(msg["content"], str) and msg["content"].strip(), \
                f"Empty content in {r['id']} role={msg['role']}"


def _check_assistant_format(records):
    for r in records:
        assistant = r["messages"][2]["content"].strip()

        # no placeholder reasoning blocks in first SFT baseline
        assert "<think>" not in assistant and "</think>" not in assistant, \
            f"Unexpected <think> block in {r['id']}"

        # exactly one \boxed{} in the whole message
        all_boxed = re.findall(r"\\boxed\{", assistant)
        assert len(all_boxed) == 1, \
            f"Expected 1 \\boxed{{}}, found {len(all_boxed)} in {r['id']}"

        # entire assistant message should be exactly the boxed expression
        boxed_expr = last_boxed_only_string(assistant)
        assert boxed_expr is not None, f"No boxed expression in {r['id']}"
        assert assistant == boxed_expr, \
            f"Assistant message should be exactly boxed answer in {r['id']}: '{assistant}'"

        # boxed answer must be on the only/last line
        last_line = assistant.split("\n")[-1].strip()
        assert last_line == boxed_expr, \
            f"Last line is not exactly the boxed expression in {r['id']}: '{last_line}'"
        
        
def _check_answer_normalization(records):
    for r in records:
        assistant = r["messages"][2]["content"]
        boxed = extract_boxed_answer(assistant)
        assert boxed is not None, f"No boxed answer in {r['id']}"

        if r["task_type"] == "mcq":
            assert MCQ_ANSWER_RE.match(str(r["answer"])), \
                f"Bad MCQ answer field '{r['answer']}' in {r['id']}"
            assert MCQ_ANSWER_RE.match(boxed.strip()), \
                f"Boxed MCQ answer '{boxed}' is not A-D in {r['id']}"
            assert boxed.strip() == str(r["answer"]).strip(), \
                f"Boxed '{boxed}' != answer '{r['answer']}' in {r['id']}"
        else:
            assert str(r["answer"]).strip(), f"Empty math answer in {r['id']}"
            assert boxed.strip() == str(r["answer"]).strip(), \
                f"Boxed '{boxed}' != answer '{r['answer']}' in {r['id']}"


# ---------------------------------------------------------------------------
# Basic existence
# ---------------------------------------------------------------------------

def test_per_dataset_files_exist():
    assert len(_all_per_dataset_files()) > 0, "No per_dataset JSONL files found."


def test_mixed_files_exist():
    for split in ("train", "val"):
        assert _mixed_file(split).exists(), f"Missing: {split}_mixed.jsonl"


def test_summary_csv_exists():
    assert (SFT_DATA_DIR / "build_summary.csv").exists()


# ---------------------------------------------------------------------------
# Per-dataset file checks
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
    expected = path.stem.rsplit("_", 1)[-1]
    for r in _load_jsonl(path):
        assert r["split"] == expected, \
            f"split '{r['split']}' != filename split '{expected}' in {r['id']}"


@pytest.mark.parametrize("path", _all_per_dataset_files())
def test_per_dataset_unique_ids(path):
    ids = [r["id"] for r in _load_jsonl(path)]
    assert len(ids) == len(set(ids)), f"Duplicate IDs in {path}"


@pytest.mark.parametrize("path", _all_per_dataset_files())
def test_per_dataset_unique_source_ids(path):
    source_ids = [r["source_id"] for r in _load_jsonl(path)]
    dupes = len(source_ids) - len(set(source_ids))
    assert dupes == 0, f"{dupes} duplicate source_ids in {path}"


def test_no_empty_per_dataset_files():
    for path in _all_per_dataset_files():
        assert len(_load_jsonl(path)) > 0, f"Empty: {path.name}"


# ---------------------------------------------------------------------------
# Train/val overlap (per-dataset)
# ---------------------------------------------------------------------------

def test_no_train_val_overlap_per_dataset():
    by_key: dict = defaultdict(lambda: {"train": set(), "val": set()})
    for path in _all_per_dataset_files():
        split = path.stem.rsplit("_", 1)[-1]
        if split not in VALID_SPLITS:
            continue
        for r in _load_jsonl(path):
            by_key[(r["dataset"], r["language"])][split].add(r["source_id"])
    for (dataset, language), splits in by_key.items():
        overlap = splits["train"] & splits["val"]
        assert not overlap, \
            f"{len(overlap)} overlapping source_ids in {dataset}/{language}"


# ---------------------------------------------------------------------------
# Summary CSV consistency (per-dataset and mixed)
# ---------------------------------------------------------------------------

def test_summary_counts_match_files():
    for row in _summary_rows():
        path = Path(row["path"])
        if not path.exists():
            continue
        actual = sum(1 for line in open(path, encoding="utf-8") if line.strip())
        assert actual == int(row["n"]), \
            f"Summary says {row['n']} but {path.name} has {actual} lines"


# ---------------------------------------------------------------------------
# Mixed file checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("split", ["train", "val"])
def test_mixed_schema(split):
    path = _mixed_file(split)
    if not path.exists():
        pytest.skip()
    _check_schema(_load_jsonl(path), path)


@pytest.mark.parametrize("split", ["train", "val"])
def test_mixed_loader_ids(split):
    path = _mixed_file(split)
    if not path.exists():
        pytest.skip()
    _check_loader_id(_load_jsonl(path))


@pytest.mark.parametrize("split", ["train", "val"])
def test_mixed_messages(split):
    path = _mixed_file(split)
    if not path.exists():
        pytest.skip()
    _check_messages(_load_jsonl(path))


@pytest.mark.parametrize("split", ["train", "val"])
def test_mixed_assistant_format(split):
    path = _mixed_file(split)
    if not path.exists():
        pytest.skip()
    _check_assistant_format(_load_jsonl(path))


@pytest.mark.parametrize("split", ["train", "val"])
def test_mixed_answer_normalization(split):
    path = _mixed_file(split)
    if not path.exists():
        pytest.skip()
    _check_answer_normalization(_load_jsonl(path))


@pytest.mark.parametrize("split", ["train", "val"])
def test_mixed_split_label_consistent(split):
    path = _mixed_file(split)
    if not path.exists():
        pytest.skip()
    for r in _load_jsonl(path):
        assert r["split"] == split, \
            f"Row {r['id']} has split='{r['split']}' in {split}_mixed.jsonl"


def test_mixed_no_train_val_overlap():
    train_path = _mixed_file("train")
    val_path   = _mixed_file("val")
    if not train_path.exists() or not val_path.exists():
        pytest.skip()
    train_keys = {
        (r["dataset"], r["language"], r["source_id"])
        for r in _load_jsonl(train_path)
    }
    val_keys = {
        (r["dataset"], r["language"], r["source_id"])
        for r in _load_jsonl(val_path)
    }
    overlap = train_keys & val_keys
    assert not overlap, f"{len(overlap)} examples in both mixed train and val"


def test_mixed_counts_match_summary():
    """Mixed file line counts must match build_summary.csv."""
    summary = {
        (row["dataset"], row["split"]): int(row["n"])
        for row in _summary_rows()
        if row["dataset"] == "mixed"
    }
    for split in ("train", "val"):
        path = _mixed_file(split)
        if not path.exists():
            continue
        actual = sum(1 for line in open(path, encoding="utf-8") if line.strip())
        assert actual > 0, f"Mixed {split} is empty"
        if ("mixed", split) in summary:
            assert actual == summary[("mixed", split)], \
                f"Summary says {summary[('mixed', split)]} but {path.name} has {actual}"


# ---------------------------------------------------------------------------
# Distribution checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("split", ["train", "val"])
def test_mixed_multiple_datasets(split):
    path = _mixed_file(split)
    if not path.exists():
        pytest.skip()
    datasets = {r["dataset"] for r in _load_jsonl(path)}
    assert len(datasets) >= 3, f"Mixed {split} only has: {datasets}"


@pytest.mark.parametrize("split", ["train", "val"])
def test_mixed_multiple_languages(split):
    path = _mixed_file(split)
    if not path.exists():
        pytest.skip()
    languages = {r["language"] for r in _load_jsonl(path)}
    assert len(languages) >= 3, f"Mixed {split} only has: {languages}"


@pytest.mark.parametrize("split", ["train", "val"])
def test_mixed_no_unexpected_split_labels(split):
    path = _mixed_file(split)
    if not path.exists():
        pytest.skip()
    bad = {r["split"] for r in _load_jsonl(path)} - VALID_SPLITS
    assert not bad, f"Unexpected split labels in mixed {split}: {bad}"