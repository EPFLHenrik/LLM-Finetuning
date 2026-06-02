"""multilingual/train/build_sft_data_v2.py

Unified SFT data builder for multilingual MCQ training.

Supports three modes:
  1. thinking      : use only distilled examples from *_distilled.jsonl files
  2. non_thinking  : use only processed/raw datasets through loaders
  3. mixed         : use distilled examples for thinking + loaders for non-thinking

Main design choices:
  - Global deduplication by source_id in all modes.
  - In mixed mode, distilled examples have priority over processed examples.
  - In mixed mode, processed examples whose source_id appears in distilled data
    in ANY split are removed before sampling.
  - No train source_id may appear in val/test, across thinking and non-thinking pools.
  - Benchmark/test-set source_ids from benchmark_all.jsonl are removed before writing.
  - Mixed mode supports configurable thinking_fraction, applied per language by default.
  - Final mixed train/val files are fully randomly shuffled.
  - Assistant formatting is centralized:
        full_thinking   -> <think>teacher CoT</think> + boxed answer
        short_thinking  -> <think>"Correct answer is X" in target language</think> + boxed answer
        boxed_only      -> \\boxed{X}

Output layout:
    <output_dir>/per_dataset/<dataset>_<language>_<sft_style>_<split>.jsonl
    <output_dir>/mixed/train_mixed.jsonl
    <output_dir>/mixed/val_mixed.jsonl
    <output_dir>/mixed/include_val.jsonl       if INCLUDE val exists
    <output_dir>/build_summary.csv

Example:
    python -m multilingual.train.build_sft_data_v2 \
      --config configs/sft/test-mixed.json \
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
from multilingual.loaders.global_piqa_loader import (
    load_global_piqa_examples,
    load_global_piqa_recarved,
)
from multilingual.loaders.head_qa_loader import load_head_qa_examples
from multilingual.loaders.include_loader import load_include_examples, load_include_recarved
from multilingual.loaders.mera_loader import load_mera_examples
from multilingual.loaders.mgsm_loader import load_mgsm_examples
from multilingual.loaders.milu_loader import load_milu_examples


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_SFT_BASE_DIR = REPO / "multilingual" / "data" / "sft"
DEFAULT_BENCHMARK_PATH = REPO / "multilingual" / "data" / "benchmarks" / "benchmark_all.jsonl"

ALL_SPLITS = ("train", "val", "test")
DEBUG_MAX = 20

DEFAULT_MODE = "mixed"
DEFAULT_THINKING_FRACTION = 0.25
DEFAULT_APPLY_RATIO_PER_LANGUAGE = True

DEFAULT_THINKING_STYLE = "full_thinking"
DEFAULT_NON_THINKING_STYLE = "boxed_only"

DEFAULT_DISTILLED_VAL_FRACTION = 0.15
DEFAULT_DISTILLED_TEST_FRACTION = 0.0
DEFAULT_ENABLE_TEST_SPLIT = False

DEFAULT_MAX_PER_LANGUAGE = 5000
DEFAULT_MAX_PER_DATASET = 10000
DEFAULT_MAX_PER_LANGUAGE_MIXED = 2500
DEFAULT_MAX_PER_LANGUAGE_MIXED_VAL = 1000


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
    "cultsportqa": {
        "languages": ["chinese", "hindi"],
        "loader": "cultsportqa",
        "max_per_language": 1500,
        "max_per_dataset": 3000,
        "split_caps": {"val": 300, "test": 0},
    },
    "belebele": {
        "languages": None,
        "loader": "belebele",
        "max_per_language": 300,
        "max_per_dataset": 1500,
        "split_caps": {"val": 100, "test": 0},
    },
    "exams": {
        "languages": ["italian", "spanish"],
        "loader": "exams",
        "split_caps": {"val": 200, "test": 0},
    },
    "head_qa": {
        "languages": ["spanish", "italian"],
        "loader": "head_qa",
        "language_caps": {"spanish": 2000, "italian": 1500},
        "max_per_dataset": 3500,
        "split_caps": {"val": 300, "test": 0},
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
# Basic IO and normalization
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


def _normalize_for_dedupe(x: str) -> str:
    return re.sub(r"\s+", " ", str(x).strip())


def _fallback_source_id(
    dataset: str,
    language: str,
    task_type: str,
    prompt: str,
    answer: str = "",
) -> str:
    raw = "\n".join([
        str(dataset),
        str(language),
        str(task_type),
        _normalize_for_dedupe(prompt),
        str(answer).strip().upper(),
    ])
    return f"{dataset}_{language}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def _get_source_id_from_record(r: Dict[str, Any]) -> str:
    sid = str(r.get("source_id") or r.get("loader_id") or "").strip()
    if sid:
        return sid

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


def _dedupe_by_source_id(
    records: Sequence[Dict[str, Any]],
    label: str = "records",
) -> List[Dict[str, Any]]:
    seen: Set[str] = set()
    out: List[Dict[str, Any]] = []
    dropped = 0

    for r in records:
        sid = _get_source_id_from_record(r)
        rr = dict(r)
        rr["source_id"] = sid

        if sid in seen:
            dropped += 1
            continue

        seen.add(sid)
        out.append(rr)

    if dropped:
        print(f"  Removed {dropped} duplicate source_id examples from {label}")

    return out


def _remove_source_ids(
    records: Sequence[Dict[str, Any]],
    forbidden_ids: Set[str],
    label: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    dropped = 0

    for r in records:
        sid = _get_source_id_from_record(r)
        if sid in forbidden_ids:
            dropped += 1
            continue
        out.append(r)

    if dropped:
        print(f"  Removed {dropped} {label} examples overlapping with forbidden source_ids")

    return out


def _reindex(records: Sequence[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, r in enumerate(records):
        rr = dict(r)
        rr["id"] = f"{rr.get('dataset', 'unknown')}_{split}_{rr.get('language', 'unknown')}_{i}"
        rr["split"] = split
        out.append(rr)
    return out


# ---------------------------------------------------------------------------
# Assistant formatting
# ---------------------------------------------------------------------------

SHORT_THINK_TEMPLATES = {
    "spanish": "La respuesta correcta es {answer}.",
    "italian": "La risposta corretta è {answer}.",
    "russian": "Правильный ответ — {answer}.",
    "hindi": "सही उत्तर {answer} है।",
    "chinese": "正确答案是 {answer}。",
}


def _boxed_answer(answer: str, task_type: str = "mcq") -> str:
    ans = str(answer).strip()
    if task_type == "mcq":
        ans = ans.upper()
    return f"\\boxed{{{ans}}}"


def _short_think_sentence(language: str, answer: str) -> str:
    template = SHORT_THINK_TEMPLATES.get(language, "The correct answer is {answer}.")
    return template.format(answer=str(answer).strip().upper())


def make_assistant_message(
    answer: str,
    task_type: str,
    assistant_style: str,
    language: str,
    cot: Optional[str] = None,
) -> str:
    ans = str(answer).strip().upper() if task_type == "mcq" else str(answer).strip()
    boxed = _boxed_answer(ans, task_type=task_type)
    cot = str(cot or "").strip()

    if assistant_style == "boxed_only":
        return boxed

    if assistant_style == "short_thinking":
        short = _short_think_sentence(language, ans)
        return f"<think>\n{short}\n</think>\n\n{boxed}"

    if assistant_style == "full_thinking":
        if not cot:
            cot = _short_think_sentence(language, ans)
        return f"<think>\n{cot}\n</think>\n\n{boxed}"

    raise ValueError(f"Unknown assistant_style: {assistant_style}")


# ---------------------------------------------------------------------------
# Distilled loading
# ---------------------------------------------------------------------------

def split_distilled_answer(clean: str, gold: str) -> Tuple[str, str]:
    lines = str(clean).strip().splitlines()

    while lines and not lines[-1].strip():
        lines.pop()

    while lines and "\\boxed{" in lines[-1]:
        lines.pop()
        while lines and not lines[-1].strip():
            lines.pop()

    return "\n".join(lines).strip(), _boxed_answer(gold, task_type="mcq")


def _extract_prompt_from_distilled(r: Dict[str, Any]) -> str:
    prompt = str(r.get("prompt", ""))
    if not prompt and r.get("messages") and len(r["messages"]) >= 2:
        prompt = str(r["messages"][1].get("content", ""))
    return prompt


def _distilled_raw_dedupe_key(r: Dict[str, Any]) -> str:
    prompt = _extract_prompt_from_distilled(r)
    answer = str(r.get("answer", r.get("distill", {}).get("gold", ""))).strip().upper()
    raw = "\n".join([
        str(r.get("dataset", "")),
        str(r.get("language", "")),
        str(r.get("task_type", "mcq")),
        _normalize_for_dedupe(prompt),
        answer,
    ])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _build_distilled_sft_record(
    r: Dict[str, Any],
    split: str,
    idx: int,
    assistant_style: str,
) -> Dict[str, Any]:
    dataset = str(r.get("dataset", "unknown"))
    language = str(r.get("language", "unknown"))
    task_type = str(r.get("task_type", "mcq"))
    gold = str(r.get("answer", r.get("distill", {}).get("gold", ""))).strip().upper()
    prompt = _extract_prompt_from_distilled(r)

    source_id = str(r.get("source_id") or r.get("id") or "").strip()
    if not source_id:
        source_id = _fallback_source_id(dataset, language, task_type, prompt, gold)

    clean = str(r.get("distill", {}).get("clean", ""))
    cot, _ = split_distilled_answer(clean, gold) if clean else ("", "")

    metadata = dict(r.get("metadata", {}))
    metadata.update({
        "source_kind": "distilled",
        "sft_style": assistant_style,
        "distilled": True,
        "used_teacher": r.get("distill", {}).get("used_teacher"),
        "distill_source_file": r.get("_src_file", ""),
    })

    return {
        "id": f"{dataset}_{split}_{language}_{assistant_style}_{idx}",
        "source_id": source_id,
        "loader_id": str(r.get("id", source_id)),
        "dataset": dataset,
        "language": language,
        "split": split,
        "task_type": task_type,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {
                "role": "assistant",
                "content": make_assistant_message(
                    answer=gold,
                    task_type=task_type,
                    assistant_style=assistant_style,
                    language=language,
                    cot=cot,
                ),
            },
        ],
        "answer": gold,
        "metadata": metadata,
    }


def load_distilled_records(
    distilled_dirs: Sequence[Path],
    assistant_style: str,
    val_fraction: float,
    test_fraction: float,
    enable_test_split: bool,
    debug: bool,
) -> Dict[str, List[Dict[str, Any]]]:
    raw_records: List[Dict[str, Any]] = []

    for distilled_dir in distilled_dirs:
        if not distilled_dir.exists():
            print(f"  [WARN] distilled_dir not found: {distilled_dir}")
            continue

        for path in sorted(distilled_dir.glob("*_distilled.jsonl")):
            rows = _load_jsonl(path)
            if debug:
                rows = rows[:DEBUG_MAX]

            print(f"  Loading distilled: {path.name} ({len(rows)} examples)")
            for r in rows:
                rr = dict(r)
                rr["_src_file"] = path.name
                raw_records.append(rr)

    seen: Set[str] = set()
    deduped: List[Dict[str, Any]] = []
    conflicts: Dict[str, Set[str]] = defaultdict(set)

    for r in raw_records:
        prompt = _extract_prompt_from_distilled(r)
        answer = str(r.get("answer", r.get("distill", {}).get("gold", ""))).strip().upper()

        prompt_key = hashlib.sha1("\n".join([
            str(r.get("dataset", "")),
            str(r.get("language", "")),
            str(r.get("task_type", "mcq")),
            _normalize_for_dedupe(prompt),
        ]).encode("utf-8")).hexdigest()

        conflicts[prompt_key].add(answer)

        key = _distilled_raw_dedupe_key(r)
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    n_conflicts = sum(1 for answers in conflicts.values() if len(answers) > 1)
    if n_conflicts:
        print(f"  [WARN] {n_conflicts} distilled prompts have conflicting answers")

    if len(raw_records) > len(deduped):
        print(f"  Removed {len(raw_records) - len(deduped)} duplicate distilled prompt+answer examples")

    rng = random.Random(VAL_SEED)

    by_group: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in deduped:
        by_group[(str(r.get("dataset", "unknown")), str(r.get("language", "unknown")))].append(r)

    out: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    split_idx = {"train": 0, "val": 0, "test": 0}

    for (dataset, language), group in sorted(by_group.items()):
        rng.shuffle(group)
        n = len(group)

        n_test = (
            max(1, int(n * test_fraction))
            if enable_test_split and test_fraction > 0 and n >= 3
            else 0
        )
        n_val = max(1, int(n * val_fraction)) if val_fraction > 0 and n >= 2 else 0

        test_group = group[:n_test]
        val_group = group[n_test:n_test + n_val]
        train_group = group[n_test + n_val:]

        groups = {
            "train": train_group,
            "val": val_group,
            "test": test_group,
        }

        for split, rows in groups.items():
            for row in rows:
                out[split].append(
                    _build_distilled_sft_record(
                        row,
                        split=split,
                        idx=split_idx[split],
                        assistant_style=assistant_style,
                    )
                )
                split_idx[split] += 1

        msg = f"  {dataset}/{language}: {len(train_group)} thinking train, {len(val_group)} thinking val"
        if enable_test_split:
            msg += f", {len(test_group)} thinking test"
        print(msg)

    for split in ALL_SPLITS:
        out[split] = _dedupe_by_source_id(out[split], label=f"distilled/{split}")

    print(
        f"  Total distilled SFT: {len(out['train'])} train, {len(out['val'])} val"
        + (f", {len(out['test'])} test" if enable_test_split else "")
    )

    return out


# ---------------------------------------------------------------------------
# Processed/raw loading
# ---------------------------------------------------------------------------

def _load_examples(
    dataset: str,
    language: str,
    split: str,
    n: Optional[int],
    hf_token: str,
) -> List[Dict[str, Any]]:
    mode = str(_dataset_cfg(dataset).get("mode", "as_is"))
    loader = str(_dataset_cfg(dataset).get("loader", dataset))

    if loader == "include":
        return load_include_examples(language=language, n=n, split=split, mode=mode)

    if loader == "global_piqa":
        return load_global_piqa_examples(language=language, n=n, split=split, mode=mode)

    if loader == "mgsm":
        return load_mgsm_examples(language=language, n=n, split=split)

    if loader == "milu":
        return load_milu_examples(n=n, split=split, hf_token=hf_token)

    if loader == "ceval":
        return load_ceval_examples(n=n, split=split)

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

    raise ValueError(f"Unknown dataset/loader: dataset={dataset}, loader={loader}")


def _load_recarved(
    dataset: str,
    language: str,
    n_val: Optional[int],
    n_test: Optional[int],
    assistant_style: str,
) -> Dict[str, List[Dict[str, Any]]]:
    if dataset == "include":
        val_examples, test_examples = load_include_recarved(language, n_val=n_val, n_test=n_test)
    elif dataset == "global_piqa":
        val_examples, test_examples = load_global_piqa_recarved(language, n_val=n_val, n_test=n_test)
    else:
        raise ValueError(f"Recarve not implemented for dataset={dataset}")

    result: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}

    if val_examples:
        result["val"] = _examples_to_sft_records(val_examples, "val", assistant_style)

    if test_examples:
        result["test"] = _examples_to_sft_records(test_examples, "test", assistant_style)

    return result


def _hash_source_id_from_example(ex: Dict[str, Any]) -> str:
    return _fallback_source_id(
        dataset=str(ex.get("dataset", "unknown")),
        language=str(ex.get("language", "unknown")),
        task_type=str(ex.get("task_type", "mcq")),
        prompt=str(ex.get("prompt", "")),
        answer=str(ex.get("answer", "")),
    )


def _example_to_sft_record(
    ex: Dict[str, Any],
    split: str,
    idx: int,
    assistant_style: str,
) -> Dict[str, Any]:
    dataset = str(ex["dataset"])
    language = str(ex["language"])
    task_type = str(ex.get("task_type", "mcq"))
    answer = str(ex["answer"]).strip().upper() if task_type == "mcq" else str(ex["answer"]).strip()
    prompt = str(ex["prompt"])

    source_id = str(ex.get("source_id") or ex.get("id") or "").strip()
    if not source_id:
        source_id = _hash_source_id_from_example(ex)

    cot = ex.get("cot") or ex.get("rationale") or ex.get("reasoning")

    metadata = dict(ex.get("metadata", {}))
    metadata.update({
        "source_kind": "processed",
        "sft_style": assistant_style,
        "distilled": False,
    })

    return {
        "id": f"{dataset}_{split}_{language}_{assistant_style}_{idx}",
        "source_id": source_id,
        "loader_id": str(ex.get("id", source_id)),
        "dataset": dataset,
        "language": language,
        "split": split,
        "task_type": task_type,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {
                "role": "assistant",
                "content": make_assistant_message(
                    answer=answer,
                    task_type=task_type,
                    assistant_style=assistant_style,
                    language=language,
                    cot=str(cot).strip() if cot else None,
                ),
            },
        ],
        "answer": answer,
        "metadata": metadata,
    }


def _examples_to_sft_records(
    examples: Sequence[Dict[str, Any]],
    split: str,
    assistant_style: str,
) -> List[Dict[str, Any]]:
    return [_example_to_sft_record(ex, split, i, assistant_style) for i, ex in enumerate(examples)]


def _remove_train_overlap_with_eval(records_by_split: Dict[str, List[Dict[str, Any]]]) -> None:
    eval_ids: Set[str] = set()

    for split in ("val", "test"):
        for r in records_by_split.get(split, []):
            eval_ids.add(_get_source_id_from_record(r))

    if not eval_ids or not records_by_split.get("train"):
        return

    before = len(records_by_split["train"])
    records_by_split["train"] = [
        r for r in records_by_split["train"]
        if _get_source_id_from_record(r) not in eval_ids
    ]

    removed = before - len(records_by_split["train"])
    if removed:
        print(f"  Removed {removed} train/eval source_id overlaps")


def load_processed_records(
    assistant_style: str,
    hf_token: str,
    debug: bool,
) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}

    for dataset in DATASET_REGISTRY:
        for language in _languages(dataset):
            print(f"\n── Loading processed {dataset}/{language} ──")

            if _is_recarve(dataset):
                n_val = _split_cap(dataset, "val", DEFAULT_MAX_PER_DATASET)
                n_test = _split_cap(dataset, "test", DEFAULT_MAX_PER_DATASET)

                n_val = None if n_val >= 10_000_000 else n_val
                n_test = None if n_test >= 10_000_000 else n_test

                try:
                    split_records = _load_recarved(dataset, language, n_val, n_test, assistant_style)
                except Exception as e:
                    print(f"  [SKIP recarve] {e}")
                    continue

            else:
                split_records = {"train": [], "val": [], "test": []}
                active_splits = [s for s in ALL_SPLITS if _split_enabled(dataset, s)]

                for split in active_splits:
                    n = DEBUG_MAX if debug else None
                    print(f"  Loading {split} ...")

                    try:
                        examples = _load_examples(dataset, language, split, n, hf_token)
                    except Exception as e:
                        print(f"  [SKIP] {e}")
                        continue

                    if not examples:
                        print("  [SKIP] empty")
                        continue

                    records = _examples_to_sft_records(examples, split, assistant_style)
                    records = _dedupe_by_source_id(records, label=f"{dataset}/{language}/{split}")
                    split_records[split] = records

                _remove_train_overlap_with_eval(split_records)

            for split in ALL_SPLITS:
                rows = split_records.get(split, [])
                if rows:
                    out[split].extend(rows)

    for split in ALL_SPLITS:
        out[split] = _dedupe_by_source_id(out[split], label=f"processed/{split}")

    print(
        f"\n  Total processed SFT: {len(out['train'])} train, "
        f"{len(out['val'])} val, {len(out['test'])} test"
    )

    return out


# ---------------------------------------------------------------------------
# Cross-pool filtering and leakage prevention
# ---------------------------------------------------------------------------

def _filter_benchmark_ids_from_pools(
    pools: Dict[str, List[Dict[str, Any]]],
    benchmark_ids: Set[str],
    label: str,
) -> None:
    if not benchmark_ids:
        return

    for split in ALL_SPLITS:
        before = len(pools.get(split, []))
        pools[split] = [
            r for r in pools.get(split, [])
            if _get_source_id_from_record(r) not in benchmark_ids
        ]
        removed = before - len(pools[split])

        if removed:
            print(f"  Removed {removed} benchmark-overlap examples from {label}/{split}")


def _remove_processed_ids_seen_in_distilled_any_split(
    thinking_by_split: Dict[str, List[Dict[str, Any]]],
    non_thinking_by_split: Dict[str, List[Dict[str, Any]]],
) -> None:
    distilled_ids: Set[str] = set()

    for split in ALL_SPLITS:
        for r in thinking_by_split.get(split, []):
            distilled_ids.add(_get_source_id_from_record(r))

    if not distilled_ids:
        return

    for split in ALL_SPLITS:
        before = len(non_thinking_by_split.get(split, []))
        non_thinking_by_split[split] = [
            r for r in non_thinking_by_split.get(split, [])
            if _get_source_id_from_record(r) not in distilled_ids
        ]
        removed = before - len(non_thinking_by_split[split])

        if removed:
            print(
                f"  Removed {removed} processed/{split} examples whose source_id "
                f"appears in distilled in any split"
            )


def _remove_train_overlap_with_val_test_across_pools(
    thinking_by_split: Dict[str, List[Dict[str, Any]]],
    non_thinking_by_split: Dict[str, List[Dict[str, Any]]],
) -> None:
    eval_ids: Set[str] = set()

    for pools in (thinking_by_split, non_thinking_by_split):
        for split in ("val", "test"):
            for r in pools.get(split, []):
                eval_ids.add(_get_source_id_from_record(r))

    if not eval_ids:
        return

    for label, pools in (
        ("thinking", thinking_by_split),
        ("non-thinking", non_thinking_by_split),
    ):
        before = len(pools.get("train", []))
        pools["train"] = [
            r for r in pools.get("train", [])
            if _get_source_id_from_record(r) not in eval_ids
        ]
        removed = before - len(pools["train"])

        if removed:
            print(
                f"  Removed {removed} {label}/train examples overlapping "
                f"with val/test across all pools"
            )


def _dedupe_each_split_across_pools(
    thinking_by_split: Dict[str, List[Dict[str, Any]]],
    non_thinking_by_split: Dict[str, List[Dict[str, Any]]],
) -> None:
    for split in ALL_SPLITS:
        seen: Set[str] = set()

        new_thinking: List[Dict[str, Any]] = []
        for r in thinking_by_split.get(split, []):
            sid = _get_source_id_from_record(r)
            if sid in seen:
                continue
            seen.add(sid)
            new_thinking.append(r)

        before_non = len(non_thinking_by_split.get(split, []))
        new_non: List[Dict[str, Any]] = []
        for r in non_thinking_by_split.get(split, []):
            sid = _get_source_id_from_record(r)
            if sid in seen:
                continue
            seen.add(sid)
            new_non.append(r)

        thinking_by_split[split] = new_thinking
        non_thinking_by_split[split] = new_non

        removed_non = before_non - len(new_non)
        if removed_non:
            print(f"  Removed {removed_non} duplicate processed/{split} examples after distilled priority")


# ---------------------------------------------------------------------------
# Sampling and mixing
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
        key = str(r.get("metadata", {}).get(stratify_by, "unknown"))
        by_group[key].append(r)

    for rows in by_group.values():
        rng.shuffle(rows)

    result: List[Dict[str, Any]] = []
    used = 0
    uncapped: List[Tuple[str, List[Dict[str, Any]]]] = []

    for group, rows in by_group.items():
        if group in subject_caps:
            cap = subject_caps[group]
            if cap == 0:
                continue
            take = rows if cap is None else rows[:cap]
            result.extend(take)
            used += len(take)
        else:
            uncapped.append((group, rows))

    remaining = max(0, global_cap - used)

    if remaining > 0 and uncapped:
        per_group = max(1, remaining // len(uncapped))
        for _, rows in uncapped:
            result.extend(rows[:per_group])

    rng.shuffle(result)
    return result[:global_cap]


def _apply_dataset_language_caps(
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

    capped_all: List[Dict[str, Any]] = []

    for dataset, ds_rows in sorted(by_dataset.items()):
        cfg = _dataset_cfg(dataset)
        stratify_by = cfg.get("stratify_by")
        subject_caps = cfg.get("subject_caps", {})

        by_lang: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in ds_rows:
            by_lang[str(r.get("language", "unknown"))].append(r)

        dataset_pool: List[Dict[str, Any]] = []

        for language, lang_rows in sorted(by_lang.items()):
            cap = _max_per_language(dataset, language, max_per_language)
            sampled = _stratified_sample(lang_rows, stratify_by, subject_caps, cap, rng)
            dataset_pool.extend(sampled)

        ds_cap = _split_cap(dataset, split, max_per_dataset)
        rng.shuffle(dataset_pool)
        capped_all.extend(dataset_pool[:ds_cap])

    rng.shuffle(capped_all)
    return capped_all


def _cap_by_language(
    records: Sequence[Dict[str, Any]],
    max_per_language_mixed: Optional[int],
    rng: random.Random,
    debug: bool,
) -> List[Dict[str, Any]]:
    rows = list(records)

    if debug or max_per_language_mixed is None:
        rng.shuffle(rows)
        return rows

    by_lang: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for r in rows:
        by_lang[str(r.get("language", "unknown"))].append(r)

    capped: List[Dict[str, Any]] = []

    for language, lang_rows in sorted(by_lang.items()):
        rng.shuffle(lang_rows)
        capped.extend(lang_rows[:max_per_language_mixed])

    rng.shuffle(capped)
    return capped


def _sample_mixed_ratio_global(
    thinking: Sequence[Dict[str, Any]],
    non_thinking: Sequence[Dict[str, Any]],
    thinking_fraction: float,
    target_total: Optional[int],
    rng: random.Random,
) -> List[Dict[str, Any]]:
    t_rows = list(thinking)
    n_rows = list(non_thinking)

    rng.shuffle(t_rows)
    rng.shuffle(n_rows)

    if target_total is None:
        target_total = len(t_rows) + len(n_rows)

    n_thinking_target = int(round(target_total * thinking_fraction))
    n_non_target = target_total - n_thinking_target

    take_t = min(len(t_rows), n_thinking_target)
    take_n = min(len(n_rows), n_non_target)

    remaining = target_total - take_t - take_n

    if remaining > 0:
        extra_t = min(len(t_rows) - take_t, remaining)
        take_t += extra_t
        remaining -= extra_t

    if remaining > 0:
        extra_n = min(len(n_rows) - take_n, remaining)
        take_n += extra_n

    mixed = t_rows[:take_t] + n_rows[:take_n]
    rng.shuffle(mixed)

    return mixed


def _sample_mixed_ratio_per_language(
    thinking: Sequence[Dict[str, Any]],
    non_thinking: Sequence[Dict[str, Any]],
    thinking_fraction: float,
    max_per_language_mixed: Optional[int],
    rng: random.Random,
    debug: bool,
) -> List[Dict[str, Any]]:
    if debug:
        return _sample_mixed_ratio_global(
            thinking=thinking,
            non_thinking=non_thinking,
            thinking_fraction=thinking_fraction,
            target_total=None,
            rng=rng,
        )

    by_lang_t: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_lang_n: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for r in thinking:
        by_lang_t[str(r.get("language", "unknown"))].append(r)

    for r in non_thinking:
        by_lang_n[str(r.get("language", "unknown"))].append(r)

    languages = sorted(set(by_lang_t) | set(by_lang_n))
    mixed: List[Dict[str, Any]] = []

    for language in languages:
        t_rows = by_lang_t.get(language, [])
        n_rows = by_lang_n.get(language, [])

        rng.shuffle(t_rows)
        rng.shuffle(n_rows)

        available = len(t_rows) + len(n_rows)

        if max_per_language_mixed is None:
            target_total = available
        else:
            target_total = min(max_per_language_mixed, available)

        lang_mixed = _sample_mixed_ratio_global(
            thinking=t_rows,
            non_thinking=n_rows,
            thinking_fraction=thinking_fraction,
            target_total=target_total,
            rng=rng,
        )

        mixed.extend(lang_mixed)

        n_t = sum(
            1 for r in lang_mixed
            if r.get("metadata", {}).get("source_kind") == "distilled"
        )
        n_n = len(lang_mixed) - n_t
        print(f"  Mixed ratio {language}: {n_t} thinking, {n_n} non-thinking")

    rng.shuffle(mixed)
    return mixed


def select_records_for_split(
    split: str,
    mode: str,
    thinking_records: Sequence[Dict[str, Any]],
    non_thinking_records: Sequence[Dict[str, Any]],
    thinking_fraction: float,
    apply_ratio_per_language: bool,
    max_per_language: int,
    max_per_dataset: int,
    max_per_language_mixed: Optional[int],
    debug: bool,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    thinking_capped = _apply_dataset_language_caps(
        thinking_records,
        split=split,
        max_per_language=max_per_language,
        max_per_dataset=max_per_dataset,
        debug=debug,
        rng=rng,
    )

    non_thinking_capped = _apply_dataset_language_caps(
        non_thinking_records,
        split=split,
        max_per_language=max_per_language,
        max_per_dataset=max_per_dataset,
        debug=debug,
        rng=rng,
    )

    if mode == "thinking":
        selected = _cap_by_language(thinking_capped, max_per_language_mixed, rng, debug)

    elif mode == "non_thinking":
        selected = _cap_by_language(non_thinking_capped, max_per_language_mixed, rng, debug)

    elif mode == "mixed":
        if apply_ratio_per_language:
            selected = _sample_mixed_ratio_per_language(
                thinking=thinking_capped,
                non_thinking=non_thinking_capped,
                thinking_fraction=thinking_fraction,
                max_per_language_mixed=max_per_language_mixed,
                rng=rng,
                debug=debug,
            )
        else:
            target_total = None
            if max_per_language_mixed is not None:
                langs = {
                    str(r.get("language", "unknown"))
                    for r in list(thinking_capped) + list(non_thinking_capped)
                }
                target_total = max_per_language_mixed * max(1, len(langs))

            selected = _sample_mixed_ratio_global(
                thinking=thinking_capped,
                non_thinking=non_thinking_capped,
                thinking_fraction=thinking_fraction,
                target_total=target_total,
                rng=rng,
            )
            selected = _cap_by_language(selected, max_per_language_mixed, rng, debug)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    selected = _dedupe_by_source_id(selected, label=f"selected/{split}")
    selected = _reindex(selected, split)
    rng.shuffle(selected)

    return selected


# ---------------------------------------------------------------------------
# Benchmark filtering / contamination check
# ---------------------------------------------------------------------------

def _load_benchmark_ids(benchmark_path: Path) -> Set[str]:
    if not benchmark_path.exists():
        print(f"  [WARN] Benchmark file not found: {benchmark_path} — skipping benchmark filtering")
        return set()

    ids: Set[str] = set()

    with open(benchmark_path, encoding="utf-8") as f:
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


def _check_benchmark_contamination(output_dir: Path, benchmark_ids: Set[str]) -> None:
    if not benchmark_ids:
        return

    print(f"\n── Benchmark contamination check ({len(benchmark_ids)} benchmark IDs) ──")

    total = 0

    for path in sorted(output_dir.rglob("*.jsonl")):
        if "test" in path.name:
            continue

        contaminated = []

        for r in _load_jsonl(path):
            sid = str(r.get("source_id") or r.get("loader_id") or "").strip()
            if sid in benchmark_ids:
                contaminated.append(sid)

        if contaminated:
            print(f"  [CONTAMINATION] {path.relative_to(output_dir)}: {len(contaminated)} examples")
            for sid in contaminated[:5]:
                print(f"    - {sid}")
            if len(contaminated) > 5:
                print(f"    ... and {len(contaminated) - 5} more")
            total += len(contaminated)

    if total == 0:
        print("  [OK] No benchmark contamination found.")
    else:
        print(f"  [ERROR] Total contaminated examples: {total}")


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def _write_per_dataset_files(
    records: Sequence[Dict[str, Any]],
    per_dir: Path,
    split: str,
    overwrite: bool,
    summary_rows: List[Dict[str, Any]],
) -> None:
    by_group: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)

    for r in records:
        dataset = str(r.get("dataset", "unknown"))
        language = str(r.get("language", "unknown"))
        sft_style = str(r.get("metadata", {}).get("sft_style", "unknown_style"))
        by_group[(dataset, language, sft_style)].append(r)

    for (dataset, language, sft_style), rows in sorted(by_group.items()):
        rows = _reindex(rows, split)
        path = per_dir / f"{dataset}_{language}_{sft_style}_{split}.jsonl"

        if _write_jsonl(rows, path, overwrite):
            summary_rows.append({
                "dataset": dataset,
                "language": language,
                "sft_style": sft_style,
                "split": split,
                "n": len(rows),
                "path": str(path),
            })


def _write_summary(summary_rows: Sequence[Dict[str, Any]], output_dir: Path) -> None:
    path = output_dir / "build_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["dataset", "language", "sft_style", "split", "n", "path"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSummary: {path}")


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build(
    output_dir: Path,
    mode: str,
    distilled_dirs: Sequence[Path],
    thinking_fraction: float,
    apply_ratio_per_language: bool,
    thinking_style: str,
    non_thinking_style: str,
    distilled_val_fraction: float,
    distilled_test_fraction: float,
    enable_test_split: bool,
    max_per_language: int,
    max_per_dataset: int,
    max_per_language_mixed: Optional[int],
    max_per_language_mixed_val: int,
    debug: bool,
    overwrite: bool,
    hf_token: str,
    benchmark_path: Optional[Path],
) -> None:
    if mode not in {"thinking", "non_thinking", "mixed"}:
        raise ValueError("mode must be one of: thinking, non_thinking, mixed")

    if not (0.0 <= thinking_fraction <= 1.0):
        raise ValueError("thinking_fraction must be between 0 and 1")

    per_dir = output_dir / "per_dataset"
    mixed_dir = output_dir / "mixed"
    rng = random.Random(VAL_SEED)
    summary_rows: List[Dict[str, Any]] = []

    thinking_by_split: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    non_thinking_by_split: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}

    if mode in {"thinking", "mixed"}:
        print("\n── Loading thinking/distilled pool ──")
        thinking_by_split = load_distilled_records(
            distilled_dirs=distilled_dirs,
            assistant_style=thinking_style,
            val_fraction=distilled_val_fraction,
            test_fraction=distilled_test_fraction,
            enable_test_split=enable_test_split,
            debug=debug,
        )

    if mode in {"non_thinking", "mixed"}:
        print("\n── Loading non-thinking/processed pool ──")
        non_thinking_by_split = load_processed_records(
            assistant_style=non_thinking_style,
            hf_token=hf_token,
            debug=debug,
        )

    benchmark_ids: Set[str] = set()

    if benchmark_path is not None:
        benchmark_ids = _load_benchmark_ids(benchmark_path)

    if benchmark_ids:
        print("\n── Removing benchmark/test-set overlaps from pools ──")
        _filter_benchmark_ids_from_pools(thinking_by_split, benchmark_ids, "thinking")
        _filter_benchmark_ids_from_pools(non_thinking_by_split, benchmark_ids, "non-thinking")

    if mode == "mixed":
        print("\n── Cross-source deduplication ──")
        _remove_processed_ids_seen_in_distilled_any_split(
            thinking_by_split=thinking_by_split,
            non_thinking_by_split=non_thinking_by_split,
        )

    print("\n── Removing train/eval overlaps across all pools ──")
    _remove_train_overlap_with_val_test_across_pools(
        thinking_by_split=thinking_by_split,
        non_thinking_by_split=non_thinking_by_split,
    )

    if mode == "mixed":
        _dedupe_each_split_across_pools(
            thinking_by_split=thinking_by_split,
            non_thinking_by_split=non_thinking_by_split,
        )

    active_splits = ["train", "val"] + (["test"] if enable_test_split else [])

    for split in active_splits:
        print(f"\n── Selecting {split} split ──")

        if split == "train":
            lang_cap = max_per_language_mixed
        elif split == "val":
            lang_cap = max_per_language_mixed_val
        else:
            lang_cap = None

        selected = select_records_for_split(
            split=split,
            mode=mode,
            thinking_records=thinking_by_split.get(split, []),
            non_thinking_records=non_thinking_by_split.get(split, []),
            thinking_fraction=thinking_fraction,
            apply_ratio_per_language=apply_ratio_per_language,
            max_per_language=max_per_language,
            max_per_dataset=max_per_dataset,
            max_per_language_mixed=lang_cap,
            debug=debug,
            rng=rng,
        )

        if not selected:
            print(f"  [WARN] No selected records for split={split}")
            continue

        _write_per_dataset_files(selected, per_dir, split, overwrite, summary_rows)

        mixed_path = mixed_dir / f"{split}_mixed.jsonl"

        if _write_jsonl(selected, mixed_path, overwrite):
            summary_rows.append({
                "dataset": "mixed",
                "language": "all",
                "sft_style": mode,
                "split": split,
                "n": len(selected),
                "path": str(mixed_path),
            })

        if split == "val":
            include_val = [r for r in selected if r.get("dataset") == "include"]
            if include_val:
                include_path = mixed_dir / "include_val.jsonl"
                if _write_jsonl(include_val, include_path, overwrite):
                    summary_rows.append({
                        "dataset": "include[val]",
                        "language": "all",
                        "sft_style": mode,
                        "split": "val",
                        "n": len(include_val),
                        "path": str(include_path),
                    })

        n_thinking = sum(
            1 for r in selected
            if r.get("metadata", {}).get("source_kind") == "distilled"
        )
        n_non = len(selected) - n_thinking

        print(f"  Selected {split}: {len(selected)} total = {n_thinking} thinking + {n_non} non-thinking")

    _write_summary(summary_rows, output_dir)

    if benchmark_ids:
        _check_benchmark_contamination(output_dir, benchmark_ids)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output_name", type=str, default=None)
    parser.add_argument("--mode", type=str, default=DEFAULT_MODE, choices=["thinking", "non_thinking", "mixed"])

    parser.add_argument("--distilled_dirs", nargs="*", type=Path, default=None)
    parser.add_argument("--thinking_fraction", type=float, default=DEFAULT_THINKING_FRACTION)
    parser.add_argument("--apply_ratio_per_language", action="store_true")
    parser.add_argument("--no_apply_ratio_per_language", action="store_true")

    parser.add_argument(
        "--thinking_style",
        type=str,
        default=DEFAULT_THINKING_STYLE,
        choices=["full_thinking", "short_thinking"],
    )
    parser.add_argument(
        "--non_thinking_style",
        type=str,
        default=DEFAULT_NON_THINKING_STYLE,
        choices=["boxed_only", "short_thinking"],
    )

    parser.add_argument("--distilled_val_fraction", type=float, default=DEFAULT_DISTILLED_VAL_FRACTION)
    parser.add_argument("--distilled_test_fraction", type=float, default=DEFAULT_DISTILLED_TEST_FRACTION)
    parser.add_argument("--enable_test_split", action="store_true")

    parser.add_argument("--max_per_language", type=int, default=DEFAULT_MAX_PER_LANGUAGE)
    parser.add_argument("--max_per_dataset", type=int, default=DEFAULT_MAX_PER_DATASET)
    parser.add_argument("--max_per_language_mixed", type=int, default=DEFAULT_MAX_PER_LANGUAGE_MIXED)
    parser.add_argument("--max_per_language_mixed_val", type=int, default=DEFAULT_MAX_PER_LANGUAGE_MIXED_VAL)

    parser.add_argument("--benchmark_path", type=Path, default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--no_benchmark_check", action="store_true")

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = DEFAULT_SFT_BASE_DIR
    mode = args.mode
    distilled_dirs = args.distilled_dirs or []
    thinking_fraction = args.thinking_fraction
    apply_ratio_per_language = DEFAULT_APPLY_RATIO_PER_LANGUAGE
    thinking_style = args.thinking_style
    non_thinking_style = args.non_thinking_style
    distilled_val_fraction = args.distilled_val_fraction
    distilled_test_fraction = args.distilled_test_fraction
    enable_test_split = args.enable_test_split
    max_per_language = args.max_per_language
    max_per_dataset = args.max_per_dataset
    max_per_language_mixed = args.max_per_language_mixed if args.max_per_language_mixed > 0 else None
    max_per_language_mixed_val = args.max_per_language_mixed_val
    benchmark_path = None if args.no_benchmark_check else args.benchmark_path

    if args.apply_ratio_per_language:
        apply_ratio_per_language = True

    if args.no_apply_ratio_per_language:
        apply_ratio_per_language = False

    if args.output_name:
        output_dir = DEFAULT_SFT_BASE_DIR / args.output_name

    if args.config:
        with open(args.config, encoding="utf-8") as f:
            cfg = json.load(f)

        mode = cfg.get("mode", mode)
        thinking_fraction = cfg.get("thinking_fraction", thinking_fraction)
        apply_ratio_per_language = cfg.get("apply_ratio_per_language", apply_ratio_per_language)

        thinking_style = cfg.get(
            "thinking_style",
            cfg.get("thinking_assistant_style", thinking_style),
        )
        non_thinking_style = cfg.get(
            "non_thinking_style",
            cfg.get("non_thinking_assistant_style", non_thinking_style),
        )

        distilled_val_fraction = cfg.get("distilled_val_fraction", distilled_val_fraction)
        distilled_test_fraction = cfg.get("distilled_test_fraction", distilled_test_fraction)
        enable_test_split = cfg.get("enable_test_split", enable_test_split)

        max_per_language = cfg.get("max_per_language", max_per_language)
        max_per_dataset = cfg.get("max_per_dataset", max_per_dataset)
        max_per_language_mixed = cfg.get("max_per_language_mixed", max_per_language_mixed)
        max_per_language_mixed_val = cfg.get("max_per_language_mixed_val", max_per_language_mixed_val)

        if output_name := cfg.get("output_name"):
            output_dir = DEFAULT_SFT_BASE_DIR / output_name

        if "distilled_dirs" in cfg:
            distilled_dirs = [Path(d) for d in cfg["distilled_dirs"]]

        if "dataset_registry" in cfg:
            DATASET_REGISTRY.clear()
            DATASET_REGISTRY.update(cfg["dataset_registry"])

        if cfg.get("no_benchmark_check", False):
            benchmark_path = None
        elif "benchmark_path" in cfg:
            benchmark_path = Path(cfg["benchmark_path"])

    hf_token = os.environ.get("HF_TOKEN", "")

    print(f"Debug:          {args.debug}  Overwrite: {args.overwrite}")
    print(f"Config:         {args.config or 'default'}")
    print(f"Output dir:     {output_dir}")
    print(f"Mode:           {mode}")
    print(f"Thinking frac:  {thinking_fraction}  per_language={apply_ratio_per_language}")
    print(f"Styles:         thinking={thinking_style}, non_thinking={non_thinking_style}")
    print(
        f"Caps:           max_per_language={max_per_language}, "
        f"max_per_dataset={max_per_dataset}, "
        f"max_per_language_mixed={max_per_language_mixed}, "
        f"max_per_language_mixed_val={max_per_language_mixed_val}"
    )
    print(
        f"Distilled:      dirs={[str(d) for d in distilled_dirs]}, "
        f"val_frac={distilled_val_fraction}, test_frac={distilled_test_fraction}"
    )
    print(f"Test split:     {'enabled' if enable_test_split else 'disabled'}")
    print(f"Benchmark check:{' disabled' if benchmark_path is None else ' ' + str(benchmark_path)}")
    print("Train ordering: fully random final shuffle")

    build(
        output_dir=output_dir,
        mode=mode,
        distilled_dirs=distilled_dirs,
        thinking_fraction=thinking_fraction,
        apply_ratio_per_language=apply_ratio_per_language,
        thinking_style=thinking_style,
        non_thinking_style=non_thinking_style,
        distilled_val_fraction=distilled_val_fraction,
        distilled_test_fraction=distilled_test_fraction,
        enable_test_split=enable_test_split,
        max_per_language=max_per_language,
        max_per_dataset=max_per_dataset,
        max_per_language_mixed=max_per_language_mixed,
        max_per_language_mixed_val=max_per_language_mixed_val,
        debug=args.debug,
        overwrite=args.overwrite,
        hf_token=hf_token,
        benchmark_path=benchmark_path,
    )


if __name__ == "__main__":
    main()