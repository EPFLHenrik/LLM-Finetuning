from typing import Any, Dict, List, Optional

from datasets import get_dataset_config_names, load_dataset

from multilingual.config import TARGET_LANGUAGES
from multilingual.utils import pick_split, select_at_most


def infer_mgsm_question_key(ex: Dict[str, Any]) -> str:
    for key in ["question", "problem", "input", "query", "prompt"]:
        if key in ex:
            return key
    raise KeyError(f"Could not infer MGSM question key. Keys: {list(ex.keys())}")


def infer_mgsm_answer_key(ex: Dict[str, Any]) -> str:
    for key in ["answer_number", "answer", "target", "output", "label"]:
        if key in ex and ex[key] is not None:
            return key
    raise KeyError(f"Could not infer MGSM answer key. Keys: {list(ex.keys())}")


def normalize_mgsm_gold_answer(answer: Any) -> str:
    answer_str = str(answer).strip()

    if "####" in answer_str:
        answer_str = answer_str.split("####")[-1].strip()

    return answer_str


def format_mgsm_example(ex: Dict[str, Any], language: str, idx: int) -> Dict[str, Any]:
    question_key = infer_mgsm_question_key(ex)
    answer_key = infer_mgsm_answer_key(ex)

    question = str(ex[question_key])
    gold = normalize_mgsm_gold_answer(ex[answer_key])

    prompt = (
        f"{question}\n\n"
        "Solve the problem. Put only the final numeric answer inside \\boxed{...}."
    )

    metadata = {
        k: ex[k]
        for k in ex.keys()
        if k not in {question_key, answer_key}
    }

    return {
        "id": f"mgsm_{language}_{idx}",
        "dataset": "mgsm",
        "language": language,
        "task_type": "math",
        "prompt": prompt,
        "answer": gold,
        "metadata": metadata,
    }


def load_mgsm_examples(language: str, n: Optional[int] = None) -> List[Dict[str, Any]]:
    mgsm_configs = TARGET_LANGUAGES[language]["mgsm_configs"]

    if not mgsm_configs:
        print(f"[MGSM] No MGSM config for {language}; skipping.")
        return []

    available_configs = get_dataset_config_names("juletxara/mgsm")
    config_lookup = {c.lower(): c for c in available_configs}

    chosen_config = None
    for candidate in mgsm_configs:
        if candidate.lower() in config_lookup:
            chosen_config = config_lookup[candidate.lower()]
            break

    if chosen_config is None:
        print(f"[MGSM] No config found for {language}. Available configs: {available_configs}")
        return []

    ds = load_dataset("juletxara/mgsm", chosen_config)

    split = pick_split(ds)
    data = ds[split]

    if len(data) == 0:
        print(f"[MGSM] Empty split for {language}.")
        return []

    sampled = select_at_most(data, n)

    return [
        format_mgsm_example(sampled[i], language, i)
        for i in range(len(sampled))
    ]