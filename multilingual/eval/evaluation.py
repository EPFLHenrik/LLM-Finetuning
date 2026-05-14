"""multilingual/evaluation.py"""

import json
import re
import sys
from typing import Any, Dict, List, Optional
import pandas as pd
from tqdm.auto import tqdm
from multilingual.config import MCQ_LETTERS, REPO
from multilingual.utils import normalize_mc_answer, normalize_text_answer
sys.path.append(str(REPO))
from evaluate.extract_answer import extract_boxed_answer, is_equiv


def extract_mcq_prediction_with_fallback(output: str) -> Optional[str]:
    """
    Diagnostic MCQ extractor. NOT the official course parser.

    It is useful for determining whether the model likely knew the answer
    but failed the boxed-output format.
    """
    boxed = extract_boxed_answer(output)
    if boxed is not None:
        boxed = boxed.strip().upper()
        if boxed in MCQ_LETTERS:
            return boxed

    text = output.strip()

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = text.replace("<|im_end|>", "").strip()
    text = text.replace("<|endoftext|>", "").strip()
    text = re.sub(r"\s+", " ", text).strip()

    patterns = [
        r"(?:answer|final answer|correct answer|risposta|respuesta|réponse|antwort|ответ|答案|答)\s*[:：]?\s*([A-Z])\b",
        r"\\boxed\s*\{?\s*([A-Z])\s*\}?",
        r"\b([A-Z])\s*[\)\.:：]",
        r"^\s*([A-Z])\b",
        r"^\s*([A-Z])\s*$",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            if letter in MCQ_LETTERS:
                return letter

    return None


def score_prediction(prediction: Optional[str], gold: str, task_type: str) -> bool:
    if prediction is None:
        return False

    if task_type == "mcq":
        return normalize_mc_answer(prediction) == normalize_mc_answer(gold)

    if task_type == "math":
        return is_equiv(prediction, gold)

    return normalize_text_answer(prediction) == normalize_text_answer(gold)


def evaluate_examples(
    examples: List[Dict[str, Any]],
    dataset_name: str,
    language: str,
    model,
    tokenizer,
    max_new_tokens_by_task: Dict[str, int],
    generate_fn,
) -> pd.DataFrame:
    rows = []

    for ex in tqdm(examples, desc=f"{dataset_name}/{language}"):
        max_new_tokens = max_new_tokens_by_task[ex["task_type"]]

        gen = generate_fn(
            model=model,
            tokenizer=tokenizer,
            prompt=ex["prompt"],
            task_type=ex["task_type"],
            max_new_tokens=max_new_tokens,
        )

        output = gen["text"]

        prediction = extract_boxed_answer(output)
        missing_box = prediction is None
        correct = score_prediction(prediction, ex["answer"], ex["task_type"])

        if ex["task_type"] == "mcq":
            fallback_prediction = extract_mcq_prediction_with_fallback(output)
        else:
            fallback_prediction = prediction

        fallback_correct = score_prediction(
            fallback_prediction,
            ex["answer"],
            ex["task_type"],
        )

        rows.append({
            "id": ex["id"],
            "dataset": dataset_name,
            "language": language,
            "task_type": ex["task_type"],
            "gold": ex["answer"],

            "prediction": prediction,
            "correct": correct,
            "missing_box": missing_box,

            "fallback_prediction": fallback_prediction,
            "fallback_correct": fallback_correct,

            "n_generated_tokens": gen["n_generated_tokens"],
            "hit_max_tokens": gen["hit_max_tokens"],
            "max_new_tokens": max_new_tokens,

            "prompt": ex["prompt"],
            "output": output,
            "metadata": json.dumps(ex.get("metadata", {}), ensure_ascii=False),
        })

    df = pd.DataFrame(rows)

    if len(df) > 0:
        n_correct = int(df["correct"].sum())
        n_total = len(df)

        strict_acc = n_correct / n_total
        fallback_acc = df["fallback_correct"].mean()
        box_compliance = (~df["missing_box"]).mean()
        max_token_rate = df["hit_max_tokens"].mean()

        valid = df[~df["missing_box"]]
        valid_only_acc = valid["correct"].mean() if len(valid) > 0 else None

        print(
            f"{dataset_name}/{language}: "
            f"strict_acc = {strict_acc:.3f} ({n_correct}/{n_total}), "
            f"box_compliance = {box_compliance:.3f}, "
            f"valid_only_acc = {valid_only_acc if valid_only_acc is not None else 'NA'}, "
            f"fallback_acc = {fallback_acc:.3f}, "
            f"hit_max_tokens = {max_token_rate:.3f}"
        )
    else:
        print(f"{dataset_name}/{language}: no examples")

    return df


def summarize_results(df_all: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []

    for (dataset, language), group in df_all.groupby(["dataset", "language"]):
        valid = group[~group["missing_box"]]

        summary_rows.append({
            "dataset": dataset,
            "language": language,
            "n": len(group),
            "strict_accuracy": group["correct"].mean(),
            "box_compliance_rate": (~group["missing_box"]).mean(),
            "valid_only_accuracy": valid["correct"].mean() if len(valid) > 0 else None,
            "fallback_accuracy": group["fallback_correct"].mean(),
            "hit_max_tokens_rate": group["hit_max_tokens"].mean(),
            "avg_generated_tokens": group["n_generated_tokens"].mean(),
            "n_valid_boxed": len(valid),
        })

    return pd.DataFrame(summary_rows)