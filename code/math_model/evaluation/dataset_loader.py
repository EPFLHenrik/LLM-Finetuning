"""Math benchmark loaders.

Every loader returns List[Dict] with the schema {"prompt": str, "answer": str},
matching validation_samples/math.jsonl. The rest of the pipeline is
benchmark-agnostic.
"""
import json
from pathlib import Path
from typing import List, Dict, Optional


def load_jsonl(path: Path) -> List[Dict]:
    items: List[Dict] = []
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
    items: List[Dict] = []
    for ex in ds:
        # GSM8K answer format: "<reasoning>\n#### 42"  — keep only the number.
        gold = ex["answer"].split("####")[-1].strip().replace(",", "")
        items.append({"prompt": ex["question"], "answer": gold})
    return items[:limit] if limit else items


def load_math500(limit: Optional[int] = None) -> List[Dict]:
    """MATH-500 — Hendrycks' MATH test set, 500 problems. Standard hard benchmark."""
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    items = [{"prompt": ex["problem"], "answer": ex["answer"]} for ex in ds]
    return items[:limit] if limit else items


def load_math500_stratified(
    n_per_level: int = 20, seed: int = 42
) -> List[Dict]:
    """MATH-500 with `n_per_level` problems per difficulty level (1..5).

    The default subset has n_per_level*5 = 100 problems balanced across levels.
    Use this for fast iteration; the first-100 of the unstratified split is
    skewed toward whatever level happens to come first in the source ordering.
    """
    import random

    from datasets import load_dataset

    rng = random.Random(seed)
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")

    by_level: Dict[int, List[Dict]] = {}
    for ex in ds:
        by_level.setdefault(int(ex["level"]), []).append(ex)

    items: List[Dict] = []
    for level in sorted(by_level):
        problems = list(by_level[level])
        rng.shuffle(problems)
        for ex in problems[:n_per_level]:
            items.append({"prompt": ex["problem"], "answer": ex["answer"]})
    return items


def load_math500_hard(limit: Optional[int] = None, seed: int = 42) -> List[Dict]:
    """MATH-500 level-5 only — the hardest tier (~135 problems).

    MATH500 at large saturates around 90% pass@8 for any halfway-decent
    finetune. Level-5 alone is much more discriminative.
    """
    import random

    from datasets import load_dataset

    rng = random.Random(seed)
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    level5 = [ex for ex in ds if int(ex["level"]) == 5]
    rng.shuffle(level5)
    items = [{"prompt": ex["problem"], "answer": ex["answer"]} for ex in level5]
    return items[:limit] if limit else items


def _aime_normalise(ex: Dict) -> Dict:
    """AIME datasets have inconsistent field names across HF mirrors.

    Try the most common variants in order: 'Problem'/'Answer' (Maxwell-Jia),
    'problem'/'answer', 'question'/'answer'.
    """
    for prompt_key in ("Problem", "problem", "question"):
        if prompt_key in ex:
            problem = ex[prompt_key]
            break
    else:
        raise KeyError(f"AIME row has no problem field: {list(ex.keys())}")
    for ans_key in ("Answer", "answer", "solution"):
        if ans_key in ex:
            ans = ex[ans_key]
            break
    else:
        raise KeyError(f"AIME row has no answer field: {list(ex.keys())}")
    # AIME answers are integers 0..999. Coerce to a clean string for Math-Verify.
    return {"prompt": problem, "answer": str(ans).strip()}


def load_aime_2024(limit: Optional[int] = None) -> List[Dict]:
    """AIME 2024 — 30 problems. Very hard competition math, integer answers 0-999."""
    from datasets import load_dataset

    ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    items = [_aime_normalise(ex) for ex in ds]
    return items[:limit] if limit else items


def load_aime_2025(limit: Optional[int] = None) -> List[Dict]:
    """AIME 2025 — 30 problems. Try MathArena first, fall back to opencompass."""
    from datasets import load_dataset

    last_err: Optional[Exception] = None
    for repo_id, split in [
        ("MathArena/aime_2025", "train"),
        ("opencompass/AIME2025", "test"),
        ("yentinglin/aime_2025", "train"),
    ]:
        try:
            ds = load_dataset(repo_id, split=split)
            items = [_aime_normalise(ex) for ex in ds]
            return items[:limit] if limit else items
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise RuntimeError(f"All AIME 2025 sources failed: {last_err}")


def load_olympiad_math(
    limit: Optional[int] = None, seed: int = 42,
    config: str = "OE_TO_maths_en_COMP",
) -> List[Dict]:
    """OlympiadBench math subset (English, open-ended, text-only).

    `Hothan/OlympiadBench` has many configs; the default here is the
    English math competition open-ended split — the right shape for
    \\boxed{} extraction.
    """
    import random

    from datasets import load_dataset

    rng = random.Random(seed)
    ds = load_dataset("Hothan/OlympiadBench", config, split="train")
    rows = list(ds)
    rng.shuffle(rows)

    items: List[Dict] = []
    for ex in rows:
        # OlympiadBench: 'question' -> problem, 'final_answer' -> List[str] gold(s).
        problem = ex.get("question") or ex.get("problem")
        gold = ex.get("final_answer") or ex.get("answer")
        if isinstance(gold, list):
            if not gold:
                continue
            gold = gold[0]
        if problem is None or gold is None:
            continue
        items.append({"prompt": str(problem), "answer": str(gold).strip()})
        if limit and len(items) >= limit:
            break
    return items


def load_minerva_math(limit: Optional[int] = None, seed: int = 42) -> List[Dict]:
    """Minerva-Math style: university-level word problems.

    `EleutherAI/hendrycks_math` is the original MATH dataset (12.5K problems);
    we take a level-4+5 sample from any subject as a Minerva proxy. Not
    identical to the paper's bespoke set, but close in spirit and difficulty.
    """
    import random

    from datasets import load_dataset

    rng = random.Random(seed)
    try:
        ds = load_dataset("EleutherAI/hendrycks_math", "algebra", split="test")
    except Exception:
        # Fallback: try the lighteval mirror, which keeps all subjects.
        ds = load_dataset("lighteval/MATH", split="test")

    hard = [ex for ex in ds if "Level 4" in ex.get("level", "")
            or "Level 5" in ex.get("level", "")]
    rng.shuffle(hard)

    items: List[Dict] = []
    for ex in hard:
        problem = ex.get("problem")
        sol = ex.get("solution", "")
        # MATH solutions end with \\boxed{...}; reuse the same extractor as the CI.
        from .extract_answer import extract_boxed_answer
        ans = extract_boxed_answer(sol, strip_double_curly_brace=True)
        if not problem or not ans:
            continue
        items.append({"prompt": problem, "answer": ans.strip()})
        if limit and len(items) >= limit:
            break
    return items