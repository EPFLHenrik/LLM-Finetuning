"""Math benchmark loaders.

Every loader returns List[Dict] with the schema {"prompt": str, "answer": str},
matching validation_samples/math.jsonl. The rest of the pipeline is
benchmark-agnostic.
"""
import json
from pathlib import Path
from typing import List, Dict, Optional


def load_jsonl(path: Path) -> List[Dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_validation_samples(repo_root: Path) -> List[Dict]:
    """The frozen 10-problem smoke test from validation_samples/math.jsonl."""
    return load_jsonl(repo_root / "validation_samples" / "math.jsonl")


def load_gsm8k(split: str = "test", limit: Optional[int] = None) -> List[Dict]:
    """GSM8K — grade-school word problems. Easy. Sanity check for the pipeline."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    items = []
    for ex in ds:
        # GSM8K answer format: "<reasoning>\n#### 42"  — keep only the number.
        gold = ex["answer"].split("####")[-1].strip().replace(",", "")
        items.append({"prompt": ex["question"], "answer": gold})
    return items[:limit] if limit else items


def load_math500(limit: Optional[int] = None) -> List[Dict]:
    """MATH-500 — Hendrycks' MATH test set, 500 problems. Hard. Standard benchmark."""
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    items = [{"prompt": ex["problem"], "answer": ex["answer"]} for ex in ds]
    return items[:limit] if limit else items