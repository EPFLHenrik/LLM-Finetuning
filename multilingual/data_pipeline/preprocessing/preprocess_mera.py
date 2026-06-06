"""Preprocess MERA into clean Russian MCQA JSONL files."""

from pathlib import Path
import json
import re
from typing import Any

import pandas as pd
from datasets import load_dataset


REPO_ROOT = Path("/scratch/workspace/emma/standard-project-m2-mystery-machine")
PROCESSED_ROOT = REPO_ROOT / "multilingual" / "data" / "processed" / "mera"

HF_DATASET = "MERA-evaluation/MERA"
SELECTED_TASKS = ["mamuramu", "mathlogicqa", "ruopenbookqa", "ruworldtree", "parus"]
DROPPED_TASKS = ["use", "rcb", "rwsd"]
ALLOWED_SPLITS = {"train", "validation", "dev"}
KEEP_TASK_TYPES = {"mcq", "numeric", "binary_choice"}
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CORRUPTION_RE = re.compile(r"[\ufffd\x00-\x08\x0b\x0c\x0e-\x1f]")


def has_corruption(x: Any) -> bool:
    return bool(CORRUPTION_RE.search(str(x)))


def clean_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def to_json_str(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False)


def format_abcd(question: str, options: dict[str, Any]) -> str:
    return (
        f"{question}\n\n"
        f"A) {clean_text(options.get('A'))}\n"
        f"B) {clean_text(options.get('B'))}\n"
        f"C) {clean_text(options.get('C'))}\n"
        f"D) {clean_text(options.get('D'))}"
    )


def make_record(subset: str, split: str, source_id: str, task_type: str, prompt: str, answer: str, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": "mera",
        "subset": subset.lower(),
        "language": "russian",
        "split": split,
        "source_id": source_id,
        "task_type": task_type,
        "prompt": clean_text(prompt),
        "answer": clean_text(answer),
        "metadata": metadata,
    }


def is_valid_record(record: dict[str, Any]) -> bool:
    metadata = record.get("metadata", {})
    metadata_str = to_json_str(metadata)
    return (
        record["split"] in {"train", "val"}
        and record["prompt"] != ""
        and record["answer"] != ""
        and isinstance(metadata, dict)
        and not any(has_corruption(x) for x in [record["prompt"], record["answer"]])
        and not has_corruption(metadata_str)
    )


def format_mamuramu(ex: dict[str, Any], split: str, idx: int) -> dict[str, Any]:
    inp = ex["inputs"]
    meta = ex.get("meta", {})
    options = {"A": inp.get("option_a"), "B": inp.get("option_b"), "C": inp.get("option_c"), "D": inp.get("option_d")}
    return make_record(
        subset="MaMuRAMu",
        split=split,
        source_id=f"MaMuRAMu_{split}_{idx}",
        task_type="mcq",
        prompt=format_abcd(clean_text(inp.get("text")), options),
        answer=clean_text(ex["outputs"]).upper(),
        metadata={"subject": clean_text(inp.get("subject")), "domain": clean_text(meta.get("domain")), "id": meta.get("id", idx)},
    )


def format_mathlogicqa(ex: dict[str, Any], split: str, idx: int) -> dict[str, Any]:
    inp = ex["inputs"]
    meta = ex.get("meta", {})
    options = {"A": inp.get("option_a"), "B": inp.get("option_b"), "C": inp.get("option_c"), "D": inp.get("option_d")}
    return make_record(
        subset="MathLogicQA",
        split=split,
        source_id=f"MathLogicQA_{split}_{idx}",
        task_type="mcq",
        prompt=format_abcd(clean_text(inp.get("text")), options),
        answer=clean_text(ex["outputs"]).upper(),
        metadata={"task": clean_text(meta.get("task")), "id": meta.get("id", idx)},
    )


def format_ruopenbookqa(ex: dict[str, Any], split: str, idx: int) -> dict[str, Any]:
    inp = ex["inputs"]
    meta = ex.get("meta", {})
    options = {"A": inp.get("option_a"), "B": inp.get("option_b"), "C": inp.get("option_c"), "D": inp.get("option_d")}
    return make_record(
        subset="ruOpenBookQA",
        split=split,
        source_id=f"ruOpenBookQA_{split}_{idx}",
        task_type="mcq",
        prompt=format_abcd(clean_text(inp.get("text") or inp.get("question")), options),
        answer=clean_text(ex["outputs"]).upper(),
        metadata={"id": meta.get("id", idx)},
    )


def format_ruworldtree(ex: dict[str, Any], split: str, idx: int) -> dict[str, Any]:
    inp = ex["inputs"]
    meta = ex.get("meta", {})
    options = {"A": inp.get("option_a"), "B": inp.get("option_b"), "C": inp.get("option_c"), "D": inp.get("option_d")}
    return make_record(
        subset="ruWorldTree",
        split=split,
        source_id=f"ruWorldTree_{split}_{idx}",
        task_type="mcq",
        prompt=format_abcd(clean_text(inp.get("question") or inp.get("text")), options),
        answer=clean_text(ex["outputs"]).upper(),
        metadata={
            "id": meta.get("id", idx),
            "exam_name": clean_text(meta.get("exam_name")),
            "school_grade": meta.get("school_grade", None),
            "knowledge_type": clean_text(meta.get("knowledge_type")),
        },
    )


def format_parus(ex: dict[str, Any], split: str, idx: int) -> dict[str, Any]:
    inp = ex["inputs"]
    meta = ex.get("meta", {})
    q_type = clean_text(meta.get("task"))
    if q_type == "cause":
        q_line = "Что является причиной описанной ситуации?"
    elif q_type == "effect":
        q_line = "Что является следствием описанной ситуации?"
    else:
        q_line = "Какой вариант наиболее правдоподобен?"
    raw_answer = clean_text(ex["outputs"])
    answer_map = {"1": "A", "2": "B"}
    if raw_answer not in answer_map:
        raise ValueError(f"Unexpected PARus answer: {raw_answer!r}")
    prompt = (
        f"{clean_text(inp.get('premise'))}\n\n"
        f"{q_line}\n\n"
        f"A) {clean_text(inp.get('choice1'))}\n"
        f"B) {clean_text(inp.get('choice2'))}"
    )
    return make_record(
        subset="parus",
        split=split,
        source_id=f"parus_{split}_{idx}",
        task_type="mcq",
        prompt=prompt,
        answer=answer_map[raw_answer],
        metadata={
            "task": q_type,
            "id": meta.get("id", idx),
            "raw_answer": raw_answer,
            "option_a": clean_text(inp.get("choice1")),
            "option_b": clean_text(inp.get("choice2")),
        },
    )


FORMATTERS = {
    "mamuramu": format_mamuramu,
    "mathlogicqa": format_mathlogicqa,
    "ruopenbookqa": format_ruopenbookqa,
    "ruworldtree": format_ruworldtree,
    "parus": format_parus,
}


def build_records():
    rows = []
    rejected_rows = []
    for task in SELECTED_TASKS:
        assert task not in DROPPED_TASKS
        ds = load_dataset(HF_DATASET, task)
        formatter = FORMATTERS[task]
        for raw_split in ds.keys():
            if raw_split not in ALLOWED_SPLITS:
                continue
            out_split = "val" if raw_split in {"validation", "dev"} else "train"
            for idx, ex in enumerate(ds[raw_split]):
                try:
                    record = formatter(ex, out_split, idx)
                    if is_valid_record(record):
                        rows.append(record)
                    else:
                        rejected_rows.append({
                            "subset": task,
                            "split": out_split,
                            "source_id": record.get("source_id", f"{task}_{out_split}_{idx}"),
                            "reason": "invalid_record",
                            "raw": to_json_str(ex),
                            "record": to_json_str(record),
                        })
                except Exception as exc:
                    rejected_rows.append({
                        "subset": task,
                        "split": out_split,
                        "source_id": f"{task}_{out_split}_{idx}",
                        "reason": repr(exc),
                        "raw": to_json_str(ex),
                        "record": "",
                    })
    clean_raw = pd.DataFrame(rows)
    clean = clean_raw[clean_raw["task_type"].isin(KEEP_TASK_TYPES)].copy()
    return clean, pd.DataFrame(rejected_rows)


def validate_record(record):
    assert isinstance(record["metadata"], dict)
    assert record["dataset"] == "mera"
    assert record["language"] == "russian"
    assert record["task_type"] == "mcq"
    assert record["answer"] in {"A", "B", "C", "D"}
    assert record["prompt"].strip()


def write_jsonl_records(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            validate_record(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    clean, _ = build_records()
    records = clean.to_dict(orient="records")
    write_jsonl_records(records, PROCESSED_ROOT / "mera.jsonl")
    for subset in sorted(clean["subset"].unique()):
        subset_records = clean[clean["subset"] == subset].to_dict(orient="records")
        write_jsonl_records(subset_records, PROCESSED_ROOT / f"mera_clean_{subset}.jsonl")
    print(f"MERA: wrote {len(records)} records to {PROCESSED_ROOT}")


if __name__ == "__main__":
    main()
