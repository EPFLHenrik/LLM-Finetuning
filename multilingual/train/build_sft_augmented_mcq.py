"""multilingual/train/build_sft_augmented_mcq.py

SFT data builder with variable-k MCQ distractor augmentation.
No distillation/thinking support — boxed_only non-thinking only.

Augmentation rules:
  - Every train example gets exactly one augmented variant.
  - k = number of EXTRA distractors added on top of existing options.
  - k is sampled uniformly from [0, MAX_TOTAL_OPTIONS - n_original_options].
  - MAX_TOTAL_OPTIONS = 20 (hard cap); adjusted per-question based on original count.
  - k=0 means shuffle gold position only (no new distractors added).
  - All original wrong options are always kept; only k new ones are added from pool.
  - Forbidden set when sampling new distractors = all existing option norms + pool gold norms.
  - Hierarchical pool lookup: (dataset, lang, subject) → (dataset, lang) → (lang,) optional.
  - Gold answers from other examples excluded as distractors (normalized match).
  - Pools built from full source datasets, benchmark/eval examples excluded.
  - Augmentation never applied to val or test splits.
  - Output: sft/augmented/<output_name>/

Usage:
    python -m multilingual.train.build_sft_augmented_mcq \
      --config configs/sft/v19_15k.json \
      --overwrite
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
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from multilingual.config import REPO, SYSTEM_PROMPT, TARGET_LANGUAGES, VAL_SEED

from multilingual.loaders.belebele_loader import load_belebele_examples
from multilingual.loaders.ceval_loader import load_ceval_examples
from multilingual.loaders.cultsportqa_loader import load_cultsportqa_examples
from multilingual.loaders.exams_loader import load_exams_examples
from multilingual.loaders.global_mmlu_loader import load_global_mmlu_examples
from multilingual.loaders.global_piqa_loader import load_global_piqa_examples, load_global_piqa_recarved
from multilingual.loaders.head_qa_loader import load_head_qa_examples
from multilingual.loaders.include_loader import load_include_examples, load_include_recarved
from multilingual.loaders.mera_loader import load_mera_examples
from multilingual.loaders.mgsm_loader import load_mgsm_examples
from multilingual.loaders.milu_loader import load_milu_examples
from multilingual.loaders.cmmlu_loader import load_cmmlu_examples
from multilingual.loaders.mmlu_prox_loader import load_mmlu_prox_examples
from multilingual.loaders.rubq_loader import load_rubq_examples
from multilingual.loaders.demre_loader import load_demre_examples
from multilingual.loaders.ccse_loader import load_ccse_examples


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_SFT_BASE_DIR = REPO / "multilingual" / "data" / "sft" / "augmented"
DEFAULT_BENCHMARK_PATH = REPO / "multilingual" / "data" / "benchmarks" / "benchmark_all.jsonl"

ALL_SPLITS = ("train", "val", "test")
DEBUG_MAX = 20

DEFAULT_STYLE = "boxed_only"
DEFAULT_MAX_PER_LANGUAGE = 5000
DEFAULT_MAX_PER_DATASET = 10000
DEFAULT_MAX_PER_LANGUAGE_MIXED = 2500
DEFAULT_MAX_PER_LANGUAGE_MIXED_VAL = 1000

MAX_TOTAL_OPTIONS = 20          # hard cap on total options per augmented question
DEFAULT_MIN_POOL_SIZE = 1       # min unique distractors needed to attempt augmentation
DEFAULT_ALLOW_CROSS_DATASET = False

# Skip augmentation entirely (non-MCQ or open-ended)
AUG_SKIP: Set[str] = {"mgsm", "rubq", "global_piqa"}


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "include": {
        "languages": None,
        "loader": "include",
        "mode": "recarve",
        "split_caps": {"train": 0, "val": None, "test": 0},
    },
    "global_piqa": {
        "languages": None,
        "loader": "global_piqa",
        "mode": "recarve",
        "split_caps": {"train": 0, "val": 300, "test": 0},
        "max_per_dataset": 2000,
    },
    "mgsm": {
        "languages": None,
        "loader": "mgsm",
        "max_per_language": 100,
        "max_per_dataset": 500,
        "split_caps": {"val": 100, "test": 0},
    },
    "milu": {
        "languages": ["hindi"],
        "loader": "milu",
        "max_per_language": 5000,
        "max_per_dataset": 5000,
        "split_caps": {"val": 400, "test": 0},
    },
    "ceval": {
        "languages": ["chinese"],
        "loader": "ceval",
        "language_caps": {"chinese": 5000},
        "max_per_dataset": 5000,
        "split_caps": {"val": 400, "test": 0},
    },
    "cmmlu": {
        "languages": ["chinese"],
        "loader": "cmmlu",
        "language_caps": {"chinese": 5000},
        "max_per_dataset": 5000,
        "split_caps": {"val": 400, "test": 0},
    },
    "cultsportqa": {
        "languages": ["chinese", "hindi"],
        "loader": "cultsportqa",
        "max_per_language": 1500,
        "max_per_dataset": 3000,
        "split_caps": {"val": 300, "test": 0},
    },
    "head_qa": {
        "languages": ["spanish", "italian"],
        "loader": "head_qa",
        "language_caps": {"spanish": 2000, "italian": 1500},
        "max_per_dataset": 3500,
        "split_caps": {"val": 300, "test": 0},
    },
    "demre": {
        "languages": ["spanish"],
        "loader": "demre",
        "max_per_language": 45,
        "max_per_dataset": 45,
        "split_caps": {"train": 45, "val": 0, "test": 0},
    },
    "ccse": {
        "languages": ["spanish"],
        "loader": "ccse",
        "max_per_language": 299,
        "max_per_dataset": 299,
        "split_caps": {"train": 299, "val": 0, "test": 0},
    },
    "mera": {
        "languages": ["russian"],
        "loader": "mera",
        "max_per_language": 3000,
        "max_per_dataset": 3000,
        "stratify_by": "subset",
        "subject_caps": {
            "mamuramu": 285,
            "ruopenbookqa": 600,
            "ruworldtree": 115,
            "mathlogicqa": 150,
            "parus": 0,
            "use": 0,
            "rcb": 0,
            "rwsd": 0,
        },
        "split_caps": {"val": 200, "test": 0},
    },
    "rubq": {
        "languages": ["russian"],
        "loader": "rubq",
        "max_per_language": 300,
        "max_per_dataset": 300,
        "split_caps": {"val": 50, "test": 0},
    },
    "global_mmlu": {
        "languages": ["russian", "chinese", "hindi", "italian", "spanish"],
        "loader": "global_mmlu",
        "language_caps": {
            "russian": 2000,
            "chinese": 800,
            "hindi": 800,
            "italian": 2000,
            "spanish": 2000,
        },
        "max_per_dataset": 5000,
        "split_caps": {"val": 600, "test": 0},
    },
    "mmlu_prox": {
        "languages": ["russian", "chinese", "hindi", "italian", "spanish"],
        "loader": "mmlu_prox",
        "max_per_language": 5000,
        "max_per_dataset": 25000,
        "split_caps": {"val": 500, "test": 0},
    },
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _dataset_cfg(dataset: str) -> Dict[str, Any]:
    return DATASET_REGISTRY.get(dataset, {})


def _languages(dataset: str) -> List[str]:
    langs = _dataset_cfg(dataset).get("languages")
    return list(langs) if langs is not None else list(TARGET_LANGUAGES.keys())


def _max_per_language(dataset: str, language: str, global_cap: int) -> int:
    cfg = _dataset_cfg(dataset)
    lang_caps = cfg.get("language_caps", {})
    if language in lang_caps:
        v = lang_caps[language]
        return v if v is not None else 10_000_000
    v = cfg.get("max_per_language", global_cap)
    return v if v is not None else 10_000_000


def _split_cap(dataset: str, split: str, global_cap: int) -> int:
    cfg = _dataset_cfg(dataset)
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
    return str(_dataset_cfg(dataset).get("mode", "as_is")) == "recarve"


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(records: Sequence[Dict[str, Any]], path: Path, overwrite: bool) -> bool:
    if path.exists() and not overwrite:
        print(f"  [SKIP] exists: {path}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records):>5} records → {path.name}")
    return True


def _normalize(x: str) -> str:
    return re.sub(r"\s+", " ", str(x)).strip().lower()


def _fallback_source_id(dataset: str, language: str, task_type: str, prompt: str, answer: str = "") -> str:
    raw = "\n".join([dataset, language, task_type, re.sub(r"\s+", " ", prompt).strip(), answer.strip().upper()])
    return f"{dataset}_{language}_{hashlib.sha1(raw.encode()).hexdigest()[:16]}"


def _get_source_id(r: Dict[str, Any]) -> str:
    sid = str(r.get("source_id") or r.get("loader_id") or "").strip()
    if sid:
        return sid
    prompt = ""
    if r.get("messages") and len(r["messages"]) >= 2:
        prompt = str(r["messages"][1].get("content", ""))
    else:
        prompt = str(r.get("prompt", ""))
    return _fallback_source_id(
        dataset=str(r.get("dataset", "unknown")),
        language=str(r.get("language", "unknown")),
        task_type=str(r.get("task_type", "mcq")),
        prompt=prompt,
        answer=str(r.get("answer", "")),
    )


def _dedupe(records: Sequence[Dict[str, Any]], label: str = "") -> List[Dict[str, Any]]:
    seen: Set[str] = set()
    out = []
    dropped = 0
    for r in records:
        sid = _get_source_id(r)
        if sid in seen:
            dropped += 1
            continue
        seen.add(sid)
        rr = dict(r)
        rr["source_id"] = sid
        out.append(rr)
    if dropped:
        print(f"  Removed {dropped} duplicates from {label}")
    return out


def _reindex(records: Sequence[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
    out = []
    for i, r in enumerate(records):
        rr = dict(r)
        rr["id"] = f"{rr.get('dataset', 'unk')}_{split}_{rr.get('language', 'unk')}_{i}"
        rr["split"] = split
        out.append(rr)
    return out


# ---------------------------------------------------------------------------
# Assistant formatting
# ---------------------------------------------------------------------------

def _make_assistant(answer: str, task_type: str) -> str:
    ans = str(answer).strip().upper() if task_type == "mcq" else str(answer).strip()
    return f"\\boxed{{{ans}}}"


# ---------------------------------------------------------------------------
# Example loading
# ---------------------------------------------------------------------------

def _load_examples(dataset: str, language: str, split: str, n: Optional[int], hf_token: str) -> List[Dict[str, Any]]:
    mode = str(_dataset_cfg(dataset).get("mode", "as_is"))
    loader = str(_dataset_cfg(dataset).get("loader", dataset))

    if loader == "include":
        return load_include_examples(language=language, n=n, split=split, mode=mode)
    if loader == "global_piqa":
        return load_global_piqa_examples(language=language, n=n, split=split, mode=mode)
    if loader == "mgsm":
        return load_mgsm_examples(language=language, n=n, split=split)
    if loader == "rubq":
        return load_rubq_examples(language=language, n=n, split=split)
    if loader == "milu":
        return load_milu_examples(n=n, split=split, hf_token=hf_token)
    if loader == "ceval":
        return load_ceval_examples(n=n, split=split)
    if loader == "cmmlu":
        return load_cmmlu_examples(language=language, n=n, split=split)
    if loader == "cultsportqa":
        return load_cultsportqa_examples(language=language, n=n, split=split)
    if loader == "belebele":
        return load_belebele_examples(language=language, n=n, split=split)
    if loader == "exams":
        return load_exams_examples(language=language, n=n, split=split)
    if loader == "head_qa":
        return load_head_qa_examples(language=language, n=n, split=split)
    if loader == "mera":
        return load_mera_examples(language=language, n=n, split=split)
    if loader == "global_mmlu":
        return load_global_mmlu_examples(language=language, n=n, split=split)
    if loader == "mmlu_prox":
        return load_mmlu_prox_examples(language=language, n=n, split=split)
    if loader == "demre":
        return load_demre_examples(language=language, n=n, split=split)
    if loader == "ccse":
        return load_ccse_examples(language=language, n=n, split=split)
    if loader == "polizia_di_stato":
        from multilingual.loaders.polizia_di_stato_loader import load_polizia_di_stato_examples
        return load_polizia_di_stato_examples(language=language, n=n, split=split)
    if loader == "quaderno_prefettizia":
        from multilingual.loaders.quaderno_prefettizia_loader import load_quaderno_prefettizia_examples
        return load_quaderno_prefettizia_examples(language=language, n=n, split=split)
    if loader == "quaderno_prefettizia_economia_politica":
        from multilingual.loaders.quaderno_prefettizia_economia_politica_loader import (
            load_quaderno_prefettizia_economia_politica_examples,
        )
        return load_quaderno_prefettizia_economia_politica_examples(language=language, n=n, split=split)
    if loader == "italian_civil_code":
        from multilingual.loaders.italian_civil_code_loader import load_italian_civil_code_examples
        return load_italian_civil_code_examples(language=language, n=n, split=split)
    if loader == "spain_boe_legislation":
        from multilingual.loaders.spain_boe_legislation_loader import load_spain_boe_legislation_examples
        return load_spain_boe_legislation_examples(language=language, n=n, split=split)
    raise ValueError(f"Unknown loader: dataset={dataset}, loader={loader}")


def _load_recarved(dataset: str, language: str, n_val: Optional[int], n_test: Optional[int]) -> Dict[str, List[Dict[str, Any]]]:
    if dataset == "include":
        val_ex, test_ex = load_include_recarved(language, n_val=n_val, n_test=n_test)
    elif dataset == "global_piqa":
        val_ex, test_ex = load_global_piqa_recarved(language, n_val=n_val, n_test=n_test)
    else:
        raise ValueError(f"Recarve not implemented: {dataset}")
    return {"train": [], "val": val_ex or [], "test": test_ex or []}


# ---------------------------------------------------------------------------
# Option parsing (shared by pool construction and augmentation)
# ---------------------------------------------------------------------------

# Matches "A. text", "A) text", "A: text" at line start
_OPTION_LINE_RE = re.compile(r"^([A-Z])[.):\s]\s*(.+)", re.MULTILINE)


def _extract_option_map(prompt: str) -> Dict[str, str]:
    """Parse option lines from a prompt → {letter: text}, insertion order."""
    return {m.group(1): m.group(2).strip() for m in _OPTION_LINE_RE.finditer(prompt)}


def _get_options(ex: Dict[str, Any]) -> List[str]:
    """
    Resolve option texts in label order (A, B, C, ...) from an example.

    Priority:
      1. ex["options"] or ex["choices"] as a list  → return as-is
      2. ex["options"] as a dict                    → sort by key A/B/C... and return values
      3. parse option lines from ex["prompt"]       → return texts in label order
    """
    raw = ex.get("options") or ex.get("choices")

    if isinstance(raw, list) and raw:
        return [str(o) for o in raw]

    if isinstance(raw, dict) and raw:
        return [str(raw[k]) for k in sorted(raw.keys())]

    # Fall back to prompt parsing
    prompt = str(ex.get("prompt", ""))
    if not prompt:
        return []
    option_map = _extract_option_map(prompt)
    if not option_map:
        return []
    # Return in A, B, C... order
    return [option_map[k] for k in sorted(option_map.keys())]


# ---------------------------------------------------------------------------
# Distractor pool
# ---------------------------------------------------------------------------

def _get_subject(ex: Dict[str, Any]) -> Optional[str]:
    return (
        ex.get("subject") or ex.get("category") or ex.get("topic")
        or ex.get("metadata", {}).get("subject")
        or ex.get("metadata", {}).get("category")
        or ex.get("metadata", {}).get("subset")
        or ex.get("metadata", {}).get("topic")
        or ex.get("subset")
    )


class DistractorPool:
    """
    Hierarchical distractor pool with three fallback levels:
      L1: (dataset, language, subject)
      L2: (dataset, language)
      L3: (language,)  — only if allow_cross_dataset=True

    Tracks:
      _distractors: pool_key → [(original_text, normalized_text)]
      _gold_norms:  pool_key → set of normalized gold strings (never used as distractors)
    """

    def __init__(self, allow_cross_dataset: bool = False) -> None:
        self.allow_cross_dataset = allow_cross_dataset
        self._distractors: Dict[Tuple, List[Tuple[str, str]]] = defaultdict(list)
        self._gold_norms: Dict[Tuple, Set[str]] = defaultdict(set)

    def _keys(self, dataset: str, language: str, subject: Optional[str]) -> List[Tuple]:
        keys = []
        if subject:
            keys.append((dataset, language, subject))
        keys.append((dataset, language))
        if self.allow_cross_dataset:
            keys.append((language,))
        return keys

    def add_example(self, ex: Dict[str, Any]) -> None:
        if ex.get("task_type", "mcq") != "mcq":
            return
        dataset = str(ex.get("dataset", "unknown"))
        language = str(ex.get("language", "unknown"))
        subject = _get_subject(ex)
        gold_letter = str(ex.get("answer", "")).strip().upper()
        options = _get_options(ex)
        if not options or not gold_letter:
            return

        gold_idx = ord(gold_letter) - ord("A")
        gold_text = options[gold_idx] if 0 <= gold_idx < len(options) else None
        gold_norm = _normalize(gold_text) if gold_text else None

        for key in self._keys(dataset, language, subject):
            if gold_norm:
                self._gold_norms[key].add(gold_norm)

        for i, opt in enumerate(options):
            if chr(ord("A") + i) == gold_letter:
                continue
            opt_str = str(opt).strip()
            if not opt_str:
                continue
            opt_norm = _normalize(opt_str)
            for key in self._keys(dataset, language, subject):
                self._distractors[key].append((opt_str, opt_norm))

    def finalize(self) -> None:
        """Deduplicate distractor lists per key, keeping first occurrence of each norm."""
        for key in list(self._distractors):
            seen: Set[str] = set()
            deduped = []
            for orig, norm in self._distractors[key]:
                if norm not in seen:
                    seen.add(norm)
                    deduped.append((orig, norm))
            self._distractors[key] = deduped

    def sample_extra(
        self,
        dataset: str,
        language: str,
        subject: Optional[str],
        forbidden_norms: Set[str],
        n: int,
        min_pool_size: int,
        rng: random.Random,
    ) -> Optional[List[str]]:
        """
        Sample n distractors using hierarchical fallback.
        forbidden_norms: normalized strings that must not appear in the result
          (includes current gold + all existing option norms + pool gold norms).
        Returns None if no level has enough candidates (>= min_pool_size after filtering).
        """
        for key in self._keys(dataset, language, subject):
            pool_forbidden = forbidden_norms | self._gold_norms.get(key, set())
            candidates: List[str] = []
            seen_norms: Set[str] = set()
            for orig, norm in self._distractors.get(key, []):
                if norm not in pool_forbidden and norm not in seen_norms:
                    seen_norms.add(norm)
                    candidates.append(orig)

            if len(candidates) >= max(n, min_pool_size):
                return rng.sample(candidates, n)

        return None


def build_pool(
    all_examples: Sequence[Dict[str, Any]],
    benchmark_ids: Set[str],
    allow_cross_dataset: bool,
    dataset_name: str = "",
) -> DistractorPool:
    pool = DistractorPool(allow_cross_dataset=allow_cross_dataset)
    skipped = 0
    n_with_options = 0
    n_without_options = 0

    for ex in all_examples:
        sid = str(ex.get("source_id") or ex.get("id") or "").strip()
        if sid and sid in benchmark_ids:
            skipped += 1
            continue
        options = _get_options(ex)
        if options:
            n_with_options += 1
        else:
            n_without_options += 1
        pool.add_example(ex)

    pool.finalize()

    n_keys = len(pool._distractors)
    n_total_distractors = sum(len(v) for v in pool._distractors.values())
    label = f"[{dataset_name}]" if dataset_name else ""
    print(
        f"  Pool {label}: {n_with_options} examples with options, "
        f"{n_without_options} without, "
        f"{n_keys} pool keys, "
        f"{n_total_distractors} unique distractors"
        + (f"  (excluded {skipped} benchmark)" if skipped else "")
    )
    return pool


def load_full_source_examples(hf_token: str, debug: bool) -> Dict[str, List[Dict[str, Any]]]:
    """Load all examples per dataset (all splits) for pool construction. Skips recarve + AUG_SKIP."""
    result: Dict[str, List[Dict[str, Any]]] = {}
    for dataset in DATASET_REGISTRY:
        if dataset in AUG_SKIP or _is_recarve(dataset):
            continue
        all_ex: List[Dict[str, Any]] = []
        for language in _languages(dataset):
            for split in ALL_SPLITS:
                n = DEBUG_MAX if debug else None
                try:
                    all_ex.extend(_load_examples(dataset, language, split, n, hf_token))
                except Exception:
                    pass
        if all_ex:
            result[dataset] = all_ex
            print(f"  Full source [{dataset}]: {len(all_ex)} examples")
    return result


# ---------------------------------------------------------------------------
# SFT record construction
# ---------------------------------------------------------------------------

def _example_to_sft_record(ex: Dict[str, Any], split: str, idx: int, style: str) -> Dict[str, Any]:
    dataset = str(ex["dataset"])
    language = str(ex["language"])
    task_type = str(ex.get("task_type", "mcq"))
    answer = str(ex["answer"]).strip().upper() if task_type == "mcq" else str(ex["answer"]).strip()
    prompt = str(ex["prompt"])

    source_id = str(ex.get("source_id") or ex.get("id") or "").strip()
    if not source_id:
        source_id = _fallback_source_id(dataset, language, task_type, prompt, answer)

    metadata = dict(ex.get("metadata", {}))
    metadata.update({"source_kind": "processed", "sft_style": style, "distilled": False})

    return {
        "id": f"{dataset}_{split}_{language}_{style}_{idx}",
        "source_id": source_id,
        "loader_id": str(ex.get("id", source_id)),
        "dataset": dataset,
        "language": language,
        "split": split,
        "task_type": task_type,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": _make_assistant(answer, task_type)},
        ],
        "answer": answer,
        "metadata": metadata,
    }


def _examples_to_sft_records(examples: Sequence[Dict[str, Any]], split: str, style: str) -> List[Dict[str, Any]]:
    return [_example_to_sft_record(ex, split, i, style) for i, ex in enumerate(examples)]


def _remove_train_eval_overlap(records_by_split: Dict[str, List[Dict[str, Any]]]) -> None:
    eval_ids = {_get_source_id(r) for s in ("val", "test") for r in records_by_split.get(s, [])}
    if not eval_ids:
        return
    before = len(records_by_split.get("train", []))
    records_by_split["train"] = [r for r in records_by_split.get("train", []) if _get_source_id(r) not in eval_ids]
    removed = before - len(records_by_split["train"])
    if removed:
        print(f"  Removed {removed} train/eval overlaps")


def load_processed_records(style: str, hf_token: str, debug: bool) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}

    for dataset in DATASET_REGISTRY:
        for language in _languages(dataset):
            print(f"\n── Loading {dataset}/{language} ──")

            if _is_recarve(dataset):
                n_val = _split_cap(dataset, "val", DEFAULT_MAX_PER_DATASET)
                n_test = _split_cap(dataset, "test", DEFAULT_MAX_PER_DATASET)
                n_val = None if n_val >= 10_000_000 else n_val
                n_test = None if n_test >= 10_000_000 else n_test
                try:
                    raw = _load_recarved(dataset, language, n_val, n_test)
                except Exception as e:
                    print(f"  [SKIP recarve] {e}")
                    continue
                split_records: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
                for split in ("val", "test"):
                    if raw.get(split):
                        split_records[split] = _examples_to_sft_records(raw[split], split, style)
            else:
                split_records = {"train": [], "val": [], "test": []}
                for split in [s for s in ALL_SPLITS if _split_enabled(dataset, s)]:
                    n = DEBUG_MAX if debug else None
                    try:
                        examples = _load_examples(dataset, language, split, n, hf_token)
                    except Exception as e:
                        print(f"  [SKIP {split}] {e}")
                        continue
                    if not examples:
                        continue
                    records = _examples_to_sft_records(examples, split, style)
                    records = _dedupe(records, f"{dataset}/{language}/{split}")
                    split_records[split] = records
                _remove_train_eval_overlap(split_records)

            for split in ALL_SPLITS:
                out[split].extend(split_records.get(split, []))

    for split in ALL_SPLITS:
        out[split] = _dedupe(out[split], f"processed/{split}")

    print(f"\n  Processed totals: {len(out['train'])} train / {len(out['val'])} val / {len(out['test'])} test")
    return out


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def _rebuild_prompt(original_prompt: str, new_options: List[str]) -> str:
    """Replace option lines in prompt with new_options."""
    lines = original_prompt.splitlines()
    first_opt = next((i for i, l in enumerate(lines) if _OPTION_LINE_RE.match(l.strip())), None)
    new_lines = [f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(new_options)]
    if first_opt is not None:
        end = first_opt
        while end < len(lines) and (_OPTION_LINE_RE.match(lines[end].strip()) or not lines[end].strip()):
            end += 1
        return "\n".join(lines[:first_opt] + new_lines + lines[end:])
    return original_prompt + "\n" + "\n".join(new_lines)


def augment_record(
    record: Dict[str, Any],
    pool: DistractorPool,
    min_pool_size: int,
    style: str,
    rng: random.Random,
) -> Dict[str, Any]:
    """
    Create one augmented variant per training record.

    k = number of EXTRA distractors added, sampled uniformly from [0, max_extra].
    max_extra = MAX_TOTAL_OPTIONS - n_original_options  (per-question, dynamic).
    k=0: shuffle gold position only, no new distractors.
    k>0: keep all original wrong options + add k new ones from pool.

    Always returns a record (falls back to shuffle-only if pool is insufficient).
    """
    task_type = str(record.get("task_type", "mcq"))
    if task_type != "mcq" or not record.get("messages") or len(record["messages"]) < 2:
        return _shuffle_gold(record, style, rng) or record

    prompt = str(record["messages"][1]["content"])
    option_map = _extract_option_map(prompt)
    n_original = len(option_map)

    if n_original == 0:
        return record

    gold_letter = str(record.get("answer", "")).strip().upper()
    gold_text = option_map.get(gold_letter)
    if not gold_text:
        return _shuffle_gold(record, style, rng) or record

    # Per-question k range: 0 to however many extras fit before hitting MAX_TOTAL_OPTIONS
    max_extra = MAX_TOTAL_OPTIONS - n_original
    if max_extra <= 0:
        # Already at or above cap — shuffle only
        return _shuffle_gold(record, style, rng) or record

    k = rng.randint(0, max_extra)

    if k == 0:
        return _shuffle_gold(record, style, rng) or record

    # Forbidden = all existing option norms (gold + wrong), to avoid sampling duplicates
    existing_norms: Set[str] = {_normalize(t) for t in option_map.values()}

    dataset = str(record.get("dataset", "unknown"))
    language = str(record.get("language", "unknown"))
    subject = (
        record.get("metadata", {}).get("subject")
        or record.get("metadata", {}).get("category")
        or record.get("metadata", {}).get("subset")
        or record.get("metadata", {}).get("topic")
    )

    extra = pool.sample_extra(
        dataset=dataset,
        language=language,
        subject=subject,
        forbidden_norms=existing_norms,
        n=k,
        min_pool_size=min_pool_size,
        rng=rng,
    )

    if extra is None:
        # Pool too small for chosen k — fall back to shuffle-only
        return _shuffle_gold(record, style, rng) or record

    # New options = all original wrong options + k new ones + gold, shuffled
    original_wrong = [text for letter, text in option_map.items() if letter != gold_letter]
    new_options = original_wrong + extra + [gold_text]
    rng.shuffle(new_options)

    # Find where gold landed after shuffle
    gold_norm = _normalize(gold_text)
    gold_idx = next(
        (i for i, opt in enumerate(new_options) if _normalize(opt) == gold_norm),
        None,
    )
    if gold_idx is None:
        return _shuffle_gold(record, style, rng) or record

    new_gold_letter = chr(ord("A") + gold_idx)
    new_prompt = _rebuild_prompt(prompt, new_options)

    aug = dict(record)
    aug["answer"] = new_gold_letter
    aug["source_id"] = record["source_id"] + f"_aug_k{k}_{new_gold_letter}"
    aug["messages"] = [
        record["messages"][0],
        {"role": "user", "content": new_prompt},
        {"role": "assistant", "content": _make_assistant(new_gold_letter, task_type)},
    ]
    aug["metadata"] = dict(record.get("metadata", {}))
    aug["metadata"].update({
        "augmented": True,
        "aug_k_extra": k,
        "aug_n_total": len(new_options),
        "aug_original_source_id": record["source_id"],
    })
    return aug


def _shuffle_gold(record: Dict[str, Any], style: str, rng: random.Random) -> Optional[Dict[str, Any]]:
    """Shuffle option order, recompute gold letter, no new distractors."""
    if not record.get("messages") or len(record["messages"]) < 2:
        return None
    prompt = str(record["messages"][1]["content"])
    gold_letter = str(record.get("answer", "")).strip().upper()
    option_map = _extract_option_map(prompt)
    gold_text = option_map.get(gold_letter)
    if not gold_text or len(option_map) < 2:
        return None

    option_texts = list(option_map.values())
    rng.shuffle(option_texts)

    gold_norm = _normalize(gold_text)
    gold_idx = next((i for i, t in enumerate(option_texts) if _normalize(t) == gold_norm), None)
    if gold_idx is None:
        return None

    new_gold_letter = chr(ord("A") + gold_idx)
    new_prompt = _rebuild_prompt(prompt, option_texts)
    task_type = str(record.get("task_type", "mcq"))

    aug = dict(record)
    aug["answer"] = new_gold_letter
    aug["source_id"] = record["source_id"] + f"_aug_shuffle_{new_gold_letter}"
    aug["messages"] = [
        record["messages"][0],
        {"role": "user", "content": new_prompt},
        {"role": "assistant", "content": _make_assistant(new_gold_letter, task_type)},
    ]
    aug["metadata"] = dict(record.get("metadata", {}))
    aug["metadata"].update({
        "augmented": True,
        "aug_k_extra": 0,
        "aug_n_total": len(option_texts),
        "aug_original_source_id": record["source_id"],
    })
    return aug


def augment_all_train(
    train_records: List[Dict[str, Any]],
    pools: Dict[str, DistractorPool],
    min_pool_size: int,
    style: str,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """
    Produce one augmented variant for every train record and append to originals.
    AUG_SKIP datasets get no augmented variant.
    All others get augment_record() which handles k=0 (shuffle-only) internally.
    """
    aug_records: List[Dict[str, Any]] = []
    k_dist: Dict[int, int] = defaultdict(int)
    n_attempted_k_gt0 = 0
    n_succeeded_k_gt0 = 0
    n_fallback_shuffle = 0
    n_skipped = 0

    for i, record in enumerate(train_records, start=1):
        if i == 1 or i % 1000 == 0 or i == len(train_records):
            print(f"  Augmenting {i}/{len(train_records)} train records...")

        dataset = str(record.get("dataset", "unknown"))
        if dataset in AUG_SKIP:
            n_skipped += 1
            continue

        aug = augment_record(
            record=record,
            pool=pools.get(dataset, DistractorPool()),
            min_pool_size=min_pool_size,
            style=style,
            rng=rng,
        )
        aug_records.append(aug)

        k_extra = aug.get("metadata", {}).get("aug_k_extra", 0)
        k_dist[k_extra] += 1

        # Track whether this was a k>0 attempt, success, or fallback
        # augment_record sets aug_k_extra=0 for both shuffle-only (k drawn=0)
        # and pool-fallback cases; we distinguish via aug_attempted_k metadata
        attempted = aug.get("metadata", {}).get("aug_attempted_k", 0)
        if attempted > 0:
            n_attempted_k_gt0 += 1
            if k_extra > 0:
                n_succeeded_k_gt0 += 1
            else:
                n_fallback_shuffle += 1

    total = len(aug_records)
    print(f"\n  Augmentation results: {total} variants generated, {n_skipped} skipped (AUG_SKIP)")
    print(f"  k>0 attempted: {n_attempted_k_gt0}  |  succeeded: {n_succeeded_k_gt0}  |  fell back to shuffle: {n_fallback_shuffle}")
    print(f"  k=0 (shuffle-only, drawn or fallback): {k_dist.get(0, 0)}")

    if total:
        buckets = [(0, 0), (1, 3), (4, 8), (9, 16), (17, 20)]
        print("  k_extra distribution:")
        for lo, hi in buckets:
            count = sum(v for k, v in k_dist.items() if lo <= k <= hi)
            pct = 100 * count // total if total else 0
            print(f"    k={lo}–{hi}: {count} ({pct}%)")

    return train_records + aug_records


# ---------------------------------------------------------------------------
# Benchmark filtering
# ---------------------------------------------------------------------------

def _load_benchmark_ids(path: Path) -> Set[str]:
    if not path.exists():
        print(f"  [WARN] Benchmark file not found: {path}")
        return set()
    ids: Set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = str(r.get("source_id") or r.get("id") or "").strip()
            if sid:
                ids.add(sid)
    return ids


def _filter_benchmark_ids(pool: Dict[str, List[Dict[str, Any]]], benchmark_ids: Set[str], label: str) -> None:
    for split in ALL_SPLITS:
        before = len(pool.get(split, []))
        pool[split] = [r for r in pool.get(split, []) if _get_source_id(r) not in benchmark_ids]
        removed = before - len(pool[split])
        if removed:
            print(f"  Removed {removed} benchmark overlaps from {label}/{split}")


def _check_contamination(output_dir: Path, benchmark_ids: Set[str]) -> None:
    if not benchmark_ids:
        return
    print(f"\n── Contamination check ({len(benchmark_ids)} benchmark IDs) ──")
    total = 0
    for path in sorted(output_dir.rglob("*.jsonl")):
        if "test" in path.name:
            continue
        hits = [
            str(r.get("source_id") or r.get("loader_id") or "")
            for r in _load_jsonl(path)
            if str(r.get("source_id") or r.get("loader_id") or "") in benchmark_ids
        ]
        if hits:
            print(f"  [CONTAMINATION] {path.relative_to(output_dir)}: {len(hits)}")
            for h in hits[:5]:
                print(f"    - {h}")
            total += len(hits)
    print("  [OK] No contamination." if total == 0 else f"  [ERROR] {total} contaminated examples")


# ---------------------------------------------------------------------------
# Sampling / capping
# ---------------------------------------------------------------------------

def _stratified_sample(
    records: Sequence[Dict[str, Any]],
    stratify_by: Optional[str],
    subject_caps: Dict[str, Optional[int]],
    global_cap: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    if not stratify_by:
        rows = list(records)
        rng.shuffle(rows)
        return rows[:global_cap]

    by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_group[str(r.get("metadata", {}).get(stratify_by, "unknown"))].append(r)
    for rows in by_group.values():
        rng.shuffle(rows)

    result: List[Dict[str, Any]] = []
    used = 0
    uncapped = []
    for group, rows in by_group.items():
        if group in subject_caps:
            cap = subject_caps[group]
            if cap == 0:
                continue
            take = rows if cap is None else rows[:cap]
            result.extend(take)
            used += len(take)
        else:
            uncapped.append(rows)

    remaining = max(0, global_cap - used)
    if remaining > 0 and uncapped:
        per_group = max(1, remaining // len(uncapped))
        for rows in uncapped:
            result.extend(rows[:per_group])

    rng.shuffle(result)
    return result[:global_cap]


def _apply_caps(
    records: Sequence[Dict[str, Any]],
    split: str,
    max_per_language: int,
    max_per_dataset: int,
    debug: bool,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    if debug:
        rows = list(records)
        rng.shuffle(rows)
        return rows

    by_dataset: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_dataset[str(r.get("dataset", "unknown"))].append(r)

    result: List[Dict[str, Any]] = []
    for dataset, ds_rows in sorted(by_dataset.items()):
        cfg = _dataset_cfg(dataset)
        by_lang: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in ds_rows:
            by_lang[str(r.get("language", "unknown"))].append(r)

        ds_pool: List[Dict[str, Any]] = []
        for language, lang_rows in sorted(by_lang.items()):
            cap = _max_per_language(dataset, language, max_per_language)
            ds_pool.extend(_stratified_sample(
                lang_rows, cfg.get("stratify_by"), cfg.get("subject_caps", {}), cap, rng,
            ))

        ds_cap = _split_cap(dataset, split, max_per_dataset)
        rng.shuffle(ds_pool)
        result.extend(ds_pool[:ds_cap])

    rng.shuffle(result)
    return result


def _cap_by_language(
    records: Sequence[Dict[str, Any]],
    max_per_language: Optional[int],
    rng: random.Random,
    debug: bool,
) -> List[Dict[str, Any]]:
    rows = list(records)
    if debug or max_per_language is None:
        rng.shuffle(rows)
        return rows

    by_lang: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_lang[str(r.get("language", "unknown"))].append(r)

    result: List[Dict[str, Any]] = []
    for lang_rows in by_lang.values():
        rng.shuffle(lang_rows)
        result.extend(lang_rows[:max_per_language])

    rng.shuffle(result)
    return result


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def _write_per_dataset(
    records: Sequence[Dict[str, Any]],
    per_dir: Path,
    split: str,
    style: str,
    overwrite: bool,
    summary: List[Dict[str, Any]],
) -> None:
    by_group: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_group[(str(r.get("dataset", "unk")), str(r.get("language", "unk")))].append(r)

    for (dataset, language), rows in sorted(by_group.items()):
        rows = _reindex(rows, split)
        path = per_dir / f"{dataset}_{language}_{style}_{split}.jsonl"
        if _write_jsonl(rows, path, overwrite):
            summary.append({"dataset": dataset, "language": language, "sft_style": style,
                            "split": split, "n": len(rows), "path": str(path)})


def _write_summary(summary: Sequence[Dict[str, Any]], output_dir: Path) -> None:
    path = output_dir / "build_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "language", "sft_style", "split", "n", "path"])
        writer.writeheader()
        writer.writerows(summary)
    print(f"\nSummary: {path}")


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build(
    output_dir: Path,
    style: str,
    max_per_language: int,
    max_per_dataset: int,
    max_per_language_mixed: Optional[int],
    max_per_language_mixed_val: int,
    min_pool_size: int,
    allow_cross_dataset: bool,
    debug: bool,
    overwrite: bool,
    hf_token: str,
    benchmark_path: Optional[Path],
) -> None:
    per_dir = output_dir / "per_dataset"
    mixed_dir = output_dir / "mixed"
    rng = random.Random(VAL_SEED)
    summary: List[Dict[str, Any]] = []

    benchmark_ids: Set[str] = set()
    if benchmark_path is not None:
        benchmark_ids = _load_benchmark_ids(benchmark_path)

    print("\n── Loading processed records ──")
    records_by_split = load_processed_records(style, hf_token, debug)

    if benchmark_ids:
        print("\n── Removing benchmark overlaps ──")
        _filter_benchmark_ids(records_by_split, benchmark_ids, "processed")

    print("\n── Loading full source examples for distractor pools ──")
    full_source = load_full_source_examples(hf_token, debug)

    print("\n── Building distractor pools ──")
    pools: Dict[str, DistractorPool] = {}
    for dataset, examples in full_source.items():
        pools[dataset] = build_pool(examples, benchmark_ids, allow_cross_dataset, dataset_name=dataset)

    print("\n── Augmenting train records (1 variant per example) ──")
    records_by_split["train"] = augment_all_train(
        train_records=records_by_split["train"],
        pools=pools,
        min_pool_size=min_pool_size,
        style=style,
        rng=rng,
    )
    records_by_split["train"] = _dedupe(records_by_split["train"], "train post-augmentation")

    for split in ["train", "val"]:
        print(f"\n── Selecting and writing {split} ──")
        lang_cap = max_per_language_mixed if split == "train" else max_per_language_mixed_val

        selected = _apply_caps(
            records_by_split.get(split, []),
            split=split,
            max_per_language=max_per_language,
            max_per_dataset=max_per_dataset,
            debug=debug,
            rng=rng,
        )
        selected = _cap_by_language(selected, lang_cap, rng, debug)
        selected = _dedupe(selected, f"selected/{split}")
        selected = _reindex(selected, split)
        rng.shuffle(selected)

        if not selected:
            print(f"  [WARN] No records for {split}")
            continue

        _write_per_dataset(selected, per_dir, split, style, overwrite, summary)

        mixed_path = mixed_dir / f"{split}_mixed.jsonl"
        if _write_jsonl(selected, mixed_path, overwrite):
            summary.append({"dataset": "mixed", "language": "all", "sft_style": style,
                            "split": split, "n": len(selected), "path": str(mixed_path)})

        if split == "val":
            include_val = [r for r in selected if r.get("dataset") == "include"]
            if include_val:
                p = mixed_dir / "include_val.jsonl"
                if _write_jsonl(include_val, p, overwrite):
                    summary.append({"dataset": "include[val]", "language": "all", "sft_style": style,
                                    "split": "val", "n": len(include_val), "path": str(p)})

        n_aug = sum(1 for r in selected if r.get("metadata", {}).get("augmented"))
        print(f"  {split}: {len(selected)} total ({n_aug} augmented)")

    _write_summary(summary, output_dir)

    if benchmark_ids:
        _check_contamination(output_dir, benchmark_ids)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--output_name", type=str, default=None)
    p.add_argument("--style", type=str, default=DEFAULT_STYLE, choices=["boxed_only", "short_thinking"])
    p.add_argument("--max_per_language", type=int, default=DEFAULT_MAX_PER_LANGUAGE)
    p.add_argument("--max_per_dataset", type=int, default=DEFAULT_MAX_PER_DATASET)
    p.add_argument("--max_per_language_mixed", type=int, default=DEFAULT_MAX_PER_LANGUAGE_MIXED)
    p.add_argument("--max_per_language_mixed_val", type=int, default=DEFAULT_MAX_PER_LANGUAGE_MIXED_VAL)
    p.add_argument("--min_pool_size", type=int, default=DEFAULT_MIN_POOL_SIZE)
    p.add_argument("--allow_cross_dataset", action="store_true")
    p.add_argument("--benchmark_path", type=Path, default=DEFAULT_BENCHMARK_PATH)
    p.add_argument("--no_benchmark_check", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = DEFAULT_SFT_BASE_DIR
    style = args.style
    max_per_language = args.max_per_language
    max_per_dataset = args.max_per_dataset
    max_per_language_mixed = args.max_per_language_mixed if args.max_per_language_mixed > 0 else None
    max_per_language_mixed_val = args.max_per_language_mixed_val
    min_pool_size = args.min_pool_size
    allow_cross_dataset = args.allow_cross_dataset
    benchmark_path = None if args.no_benchmark_check else args.benchmark_path

    if args.output_name:
        output_dir = DEFAULT_SFT_BASE_DIR / args.output_name

    if args.config:
        with open(args.config, encoding="utf-8") as f:
            cfg = json.load(f)

        style = cfg.get("style", cfg.get("non_thinking_style", style))
        max_per_language = cfg.get("max_per_language", max_per_language)
        max_per_dataset = cfg.get("max_per_dataset", max_per_dataset)
        max_per_language_mixed = cfg.get("max_per_language_mixed", max_per_language_mixed)
        max_per_language_mixed_val = cfg.get("max_per_language_mixed_val", max_per_language_mixed_val)
        min_pool_size = cfg.get("min_pool_size", min_pool_size)
        allow_cross_dataset = cfg.get("allow_cross_dataset", allow_cross_dataset)
        if output_name := cfg.get("output_name"):
            output_dir = DEFAULT_SFT_BASE_DIR / output_name
        if "dataset_registry" in cfg:
            DATASET_REGISTRY.clear()
            DATASET_REGISTRY.update(cfg["dataset_registry"])
        if cfg.get("no_benchmark_check"):
            benchmark_path = None
        elif "benchmark_path" in cfg:
            benchmark_path = Path(cfg["benchmark_path"])

    hf_token = os.environ.get("HF_TOKEN", "")

    print(f"Output dir:       {output_dir}")
    print(f"Style:            {style}")
    print(f"MAX_TOTAL_OPTIONS:{MAX_TOTAL_OPTIONS}  (k=0..MAX-n_original per question)")
    print(f"Caps:             per_language={max_per_language}, per_dataset={max_per_dataset}, "
          f"mixed_train={max_per_language_mixed}, mixed_val={max_per_language_mixed_val}")
    print(f"Pool:             min_pool_size={min_pool_size}, cross_dataset={allow_cross_dataset}")
    print(f"Aug skip:         {AUG_SKIP}")
    print(f"Benchmark check:  {'disabled' if benchmark_path is None else str(benchmark_path)}")
    print(f"Debug: {args.debug}  Overwrite: {args.overwrite}")

    build(
        output_dir=output_dir,
        style=style,
        max_per_language=max_per_language,
        max_per_dataset=max_per_dataset,
        max_per_language_mixed=max_per_language_mixed,
        max_per_language_mixed_val=max_per_language_mixed_val,
        min_pool_size=min_pool_size,
        allow_cross_dataset=allow_cross_dataset,
        debug=args.debug,
        overwrite=args.overwrite,
        hf_token=hf_token,
        benchmark_path=benchmark_path,
    )


if __name__ == "__main__":
    main()