from typing import Any, Dict, List, Optional

from datasets import load_dataset
from multilingual.config import LETTERS, TARGET_LANGUAGES
from multilingual.utils import pick_split, select_at_most


def format_include_example(ex: Dict[str, Any], language: str, idx: int) -> Dict[str, Any]:
    question = str(ex["question"])

    choices = [
        str(ex["option_a"]),
        str(ex["option_b"]),
        str(ex["option_c"]),
        str(ex["option_d"]),
    ]

    options = "\n".join(
        f"{LETTERS[i]}) {choice}"
        for i, choice in enumerate(choices)
    )

    prompt = (
        f"{question}\n\n"
        f"{options}\n\n"
        "Choose the correct option. "
        "Return exactly one answer in this format: \\boxed{X}, replacing X by the correct option letter. "
        "Do not write the option text. Do not write an explanation."
    )

    gold = LETTERS[int(ex["answer"])]

    metadata = {
        "language": ex.get("language"),
        "country": ex.get("country"),
        "domain": ex.get("domain"),
        "subject": ex.get("subject"),
        "regional_feature": ex.get("regional_feature"),
        "level": ex.get("level"),
    }

    return {
        "id": f"include_{language}_{idx}",
        "dataset": "include",
        "language": language,
        "task_type": "mcq",
        "prompt": prompt,
        "answer": gold,
        "metadata": metadata,
    }


def load_include_examples(language: str, n: Optional[int] = None) -> List[Dict[str, Any]]:
    include_config = TARGET_LANGUAGES[language]["include_config"]

    if include_config is None:
        print(f"[INCLUDE] No INCLUDE config for {language}; skipping.")
        return []

    ds = load_dataset("CohereLabs/include-base-44", include_config)

    split = "test" if "test" in ds else pick_split(ds)
    data = ds[split]

    if len(data) == 0:
        print(f"[INCLUDE] Empty split for {language}.")
        return []

    sampled = select_at_most(data, n)

    return [
        format_include_example(sampled[i], language, i)
        for i in range(len(sampled))
    ]