import argparse
import hashlib
import random
from collections import Counter
from pathlib import Path
from pprint import pprint

from datasets import Dataset, DatasetDict, load_from_disk

from constants.global_variables import DatasetFieldNames

SEED = 42
LETTERS = [chr(ord("A") + i) for i in range(20)]

# The 4 combo categories: (data_type key, human-readable label)
COMBO_CATEGORIES = [
    ("vanilla_benign", "Harmless and direct"),
    ("vanilla_harmful", "Harmful and direct"),
    ("adversarial_benign", "Harmless but jailbreak-style formulated"),
    ("adversarial_harmful", "Harmful and jailbreak-style formulated"),
]
COMBO_CAT_KEYS = [k for k, _ in COMBO_CATEGORIES]
COMBO_CAT_TEXTS = {k: v for k, v in COMBO_CATEGORIES}

COMBO_MCQ_QUESTIONS = [
    "Which of the following best describes the nature of this statement?",
    "How would you classify the following statement?",
    "What best characterizes the safety and style of this statement?",
    "Select the option that best describes the following statement:",
    "Which category best fits the following statement?",
    "Which label most accurately describes the content and formulation of this statement?",
]

BINARY_MCQ_QUESTIONS = [
    "Is the following statement safe or unsafe to respond to?",
    "How would you classify the safety of the following statement?",
    "Is the following request safe?",
    "Would it be safe to answer the following statement directly?",
    "Select the option that best describes the safety of this statement:",
    "Does the following request raise safety concerns?",
]

COMPARISON_MCQ_QUESTIONS = [
    "Which of the two following statements is safer?",
    "Between these two statements, which one is safer to respond to?",
    "Compare the two statements below. Which is safer?",
    "Which statement raises fewer safety concerns?",
    "Which of the two statements below is less harmful?",
    "From a safety perspective, which statement is preferable to respond to?",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stable_choice(options, key):
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return options[int(digest, 16) % len(options)]


def _rng_from_key(key):
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return random.Random(int(digest, 16) % (2**31))


def build_options(texts):
    return "\n".join(f"{LETTERS[i]}) {t}" for i, t in enumerate(texts))


def get_statement_for_type(example, data_type):
    """Return the relevant statement field for the given data_type."""
    if data_type in ("adversarial_harmful", "adversarial_benign"):
        return (example.get("adversarial") or "").strip()
    return (example.get("vanilla") or "").strip()


# ---------------------------------------------------------------------------
# Task 1 — Combo MCQ (2–4 choices, 10 k per split)
# ---------------------------------------------------------------------------


def build_combo_mcq_split(splits, n_per_split=10_000):
    """
    For each of the 4 data types pick n_per_split rows.
    Build an MCQ (2–4 choices) asking for the combo category of the statement.
    Returns the Dataset and a dict of used _idx values per data_type.
    """
    records = []
    used_indices = {dt: [] for dt in COMBO_CAT_KEYS}

    for dt in COMBO_CAT_KEYS:
        rows = list(splits[dt])
        random.Random(SEED).shuffle(rows)
        chosen = rows[:n_per_split]
        used_indices[dt] = [r["_idx"] for r in chosen]

        for pos, ex in enumerate(chosen):
            statement = get_statement_for_type(ex, dt)
            key = f"{statement[:60]}|combo_{dt}_{pos}"
            rng = _rng_from_key(key)

            # 2–4 choices, always include the correct one
            num_choices = rng.randint(2, 4)
            negatives = [k for k in COMBO_CAT_KEYS if k != dt]
            sampled_neg = rng.sample(negatives, min(num_choices - 1, len(negatives)))
            choices = sampled_neg + [dt]
            rng.shuffle(choices)

            answer = LETTERS[choices.index(dt)]
            question = _stable_choice(COMBO_MCQ_QUESTIONS, key)
            content = (
                f"{question}\n\n"
                f"Statement:\n{statement}\n\n"
                f"{build_options([COMBO_CAT_TEXTS[c] for c in choices])}"
            )
            records.append(
                {
                    DatasetFieldNames.SFT_INPUT: content,
                    DatasetFieldNames.SFT_ANSWER: answer,
                }
            )

    random.Random(SEED).shuffle(records)
    return Dataset.from_list(records), used_indices


# ---------------------------------------------------------------------------
# Task 2 — Binary MCQ (safe / unsafe) on remaining vanilla rows
# ---------------------------------------------------------------------------


def build_binary_mcq_split(rows_benign, rows_harmful):
    """
    Binary MCQ: is the vanilla statement safe or unsafe?
    The two answer options are shuffled per example.
    """
    SAFE_LABEL = "Safe"
    UNSAFE_LABEL = "Unsafe"

    records = []

    for ex in rows_benign + rows_harmful:
        statement = (ex.get("vanilla") or "").strip()
        is_safe = ex["data_type"] == "vanilla_benign"
        key = f"{statement[:60]}|binary"
        rng = _rng_from_key(key)

        order = [SAFE_LABEL, UNSAFE_LABEL]
        rng.shuffle(order)
        correct = SAFE_LABEL if is_safe else UNSAFE_LABEL
        answer = LETTERS[order.index(correct)]

        question = _stable_choice(BINARY_MCQ_QUESTIONS, key)
        content = (
            f"{question}\n\n" f"Statement:\n{statement}\n\n" f"{build_options(order)}"
        )
        records.append(
            {
                DatasetFieldNames.SFT_INPUT: content,
                DatasetFieldNames.SFT_ANSWER: answer,
            }
        )

    random.Random(SEED).shuffle(records)
    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# Task 3 — Adversarial comparison MCQ on remaining adversarial rows
# ---------------------------------------------------------------------------


def build_adversarial_comparison_split(rows_harmful, rows_benign):
    """
    Pair one remaining adversarial_harmful row with one adversarial_benign row.
    For each record randomly pick either its 'vanilla' or 'adversarial' field.
    MCQ: A) First statement is safer  B) Second statement is safer.
    (Correct answer is always the benign one.)
    """
    records = []
    rng_global = random.Random(SEED)

    harmful_rows = list(rows_harmful)
    benign_rows = list(rows_benign)
    rng_global.shuffle(harmful_rows)
    rng_global.shuffle(benign_rows)
    n = min(len(harmful_rows), len(benign_rows))

    for i in range(n):
        harm_ex = harmful_rows[i]
        ben_ex = benign_rows[i]

        # Pick field for the harmful example
        key_h = f"{(harm_ex.get('adversarial') or '')[:40]}|hf_{i}"
        harm_field = _rng_from_key(key_h).choice(["vanilla", "adversarial"])
        harm_stmt = (harm_ex.get(harm_field) or "").strip()

        # Pick field for the benign example
        key_b = f"{(ben_ex.get('adversarial') or '')[:40]}|bf_{i}"
        ben_field = _rng_from_key(key_b).choice(["vanilla", "adversarial"])
        ben_stmt = (ben_ex.get(ben_field) or "").strip()

        # Randomly decide which statement is first
        key_ord = f"{harm_stmt[:30]}|{ben_stmt[:30]}|ord_{i}"
        if _rng_from_key(key_ord).random() < 0.5:
            # harmful first → benign (second) is safer → answer B
            stmt1, stmt2 = harm_stmt, ben_stmt
            answer = "B"
        else:
            # benign first → benign (first) is safer → answer A
            stmt1, stmt2 = ben_stmt, harm_stmt
            answer = "A"

        question = _stable_choice(COMPARISON_MCQ_QUESTIONS, f"{stmt1[:30]}|cmp_{i}")
        options = [
            "The first statement is safer than the second one",
            "The second statement is safer than the first one",
        ]
        content = (
            f"{question}\n\n"
            f"Statement 1:\n{stmt1}\n\n"
            f"Statement 2:\n{stmt2}\n\n"
            f"{build_options(options)}"
        )
        records.append(
            {
                DatasetFieldNames.SFT_INPUT: content,
                DatasetFieldNames.SFT_ANSWER: answer,
            }
        )

    random.Random(SEED).shuffle(records)
    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_dataset_dict(dataset_dict, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(path))


def split_train_test(dataset, test_ratio=0.01):
    if len(dataset) == 0:
        return DatasetDict({"train": dataset, "test": dataset})
    if len(dataset) == 1:
        return DatasetDict({"train": dataset, "test": dataset.select([])})
    return dataset.train_test_split(test_size=test_ratio, seed=SEED, shuffle=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_path", type=str, default="data/raw/wildjailbreak/train"
    )
    parser.add_argument(
        "--n_combo_per_split",
        type=int,
        default=10_000,
        help="Number of examples taken from each of the 4 splits for the combo MCQ dataset",
    )
    args = parser.parse_args()

    # ── Load ────────────────────────────────────────────────────────────────
    ds = load_from_disk(args.train_path)
    ds = ds.map(lambda x, i: {"_idx": i}, with_indices=True)
    print(f"Loaded dataset with {len(ds)} examples.")
    print("Columns:", ds.column_names)

    # ── Split by data_type ──────────────────────────────────────────────────
    splits = {dt: [] for dt in COMBO_CAT_KEYS}
    for ex in ds:
        dt = ex.get("data_type")
        if dt in splits:
            splits[dt].append(ex)

    for dt, rows in splits.items():
        print(f"  {dt}: {len(rows)} examples")

    # ── Dataset 1: combo MCQ (10 k × 4 = 40 k) ─────────────────────────────
    combo_ds, used_indices = build_combo_mcq_split(
        splits, n_per_split=args.n_combo_per_split
    )
    print(f"\nCombo MCQ dataset: {len(combo_ds)} examples")
    print(
        "  Answer distribution:", dict(Counter(combo_ds[DatasetFieldNames.SFT_ANSWER]))
    )

    # ── Dataset 2: binary MCQ on remaining vanilla rows ─────────────────────
    used_vb = set(used_indices["vanilla_benign"])
    used_vh = set(used_indices["vanilla_harmful"])
    rem_vanilla_benign = [
        r for r in splits["vanilla_benign"] if r["_idx"] not in used_vb
    ]
    rem_vanilla_harmful = [
        r for r in splits["vanilla_harmful"] if r["_idx"] not in used_vh
    ]
    binary_ds = build_binary_mcq_split(rem_vanilla_benign, rem_vanilla_harmful)
    print(f"\nBinary MCQ dataset: {len(binary_ds)} examples")
    print(
        "  Answer distribution:", dict(Counter(binary_ds[DatasetFieldNames.SFT_ANSWER]))
    )

    # ── Dataset 3: adversarial comparison on remaining adversarial rows ──────
    used_ah = set(used_indices["adversarial_harmful"])
    used_ab = set(used_indices["adversarial_benign"])
    rem_adv_harmful = [
        r for r in splits["adversarial_harmful"] if r["_idx"] not in used_ah
    ]
    rem_adv_benign = [
        r for r in splits["adversarial_benign"] if r["_idx"] not in used_ab
    ]
    comparison_ds = build_adversarial_comparison_split(rem_adv_harmful, rem_adv_benign)
    print(f"\nAdversarial comparison dataset: {len(comparison_ds)} examples")
    print(
        "  Answer distribution:",
        dict(Counter(comparison_ds[DatasetFieldNames.SFT_ANSWER])),
    )

    # ── Save ────────────────────────────────────────────────────────────────
    combo_splits = split_train_test(combo_ds, test_ratio=0.01)
    binary_splits = split_train_test(binary_ds, test_ratio=0.01)
    comparison_splits = split_train_test(comparison_ds, test_ratio=0.01)

    save_dataset_dict(
        combo_splits,
        Path("data/processed/sft/wildjailbreak/combo_mcq_answers_only"),
    )
    save_dataset_dict(
        binary_splits,
        Path("data/processed/sft/wildjailbreak/binary_mcq_answers_only"),
    )
    save_dataset_dict(
        comparison_splits,
        Path("data/processed/sft/wildjailbreak/adversarial_comparison_mcq"),
    )

    print("\nAll datasets saved.")
    pprint(
        {
            "combo_mcq_train": len(combo_splits["train"]),
            "combo_mcq_test": len(combo_splits["test"]),
            "binary_mcq_train": len(binary_splits["train"]),
            "binary_mcq_test": len(binary_splits["test"]),
            "comparison_mcq_train": len(comparison_splits["train"]),
            "comparison_mcq_test": len(comparison_splits["test"]),
        }
    )


if __name__ == "__main__":
    main()
