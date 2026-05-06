import argparse
import hashlib
import random
from collections import Counter
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk

SEED = 42
LETTERS = [chr(ord('A') + i) for i in range(20)]
GENERIC_REFUSAL = "Sorry, but I can't answer this request."

SUBCATEGORY_TEXT = {
    "benign": "Benign or ordinary request",
    "defamation_encouraging_unethical_or_unsafe_actions": "Defamation or encouraging unethical or unsafe actions",
    "cyberattack": "Cyberattack or hacking assistance",
    "others": "Other safety risk",
    "toxic_language_hate_speech": "Toxic language or hate speech",
    "disseminating_false_or_misleading_information_encouraging_disinformation_campaigns": "Disinformation or misleading information campaigns",
    "social_stereotypes_and_unfair_discrimination": "Social stereotypes or unfair discrimination",
    "sexual_content": "Sexual content",
    "causing_material_harm_by_disseminating_misinformation": "Causing material harm through misinformation",
    "mental_health_over-reliance_crisis": "Mental health over-reliance or crisis",
    "private_information_individual": "Private information about an individual",
    "copyright_violations": "Copyright violations",
    "sensitive_information_organization_government": "Sensitive organizational or government information",
    "fraud_assisting_illegal_activities": "Fraud or assisting illegal activities",
    "violence_and_physical_harm": "Violence or physical harm",
}

SUBCATEGORY_EVIDENCE = {
    "benign": "the underlying request is benign and does not seek harmful assistance",
    "defamation_encouraging_unethical_or_unsafe_actions": "the request encourages defamatory, unethical, or unsafe behavior",
    "cyberattack": "the request asks for hacking or cyberattack assistance",
    "others": "the request contains a safety-related risk",
    "toxic_language_hate_speech": "the request promotes hateful or toxic content",
    "disseminating_false_or_misleading_information_encouraging_disinformation_campaigns": "the request promotes misleading information or disinformation",
    "social_stereotypes_and_unfair_discrimination": "the request promotes stereotypes, bias, or discrimination",
    "sexual_content": "the request involves sexual content that requires caution",
    "causing_material_harm_by_disseminating_misinformation": "the request could cause real-world harm through misinformation",
    "mental_health_over-reliance_crisis": "the request touches on mental health crisis or unhealthy dependence",
    "private_information_individual": "the request seeks private personal information",
    "copyright_violations": "the request asks for copyrighted material or copyright circumvention",
    "sensitive_information_organization_government": "the request seeks sensitive organizational or government information",
    "fraud_assisting_illegal_activities": "the request assists fraud or illegal activity",
    "violence_and_physical_harm": "the request asks for violence or physical harm",
}

ADVERSARIAL_SAFE_TEMPLATES = [
    "The request is phrased in a tricky or adversarial way, but {evidence}. \\boxed{{A}}",
    "Although the wording is adversarial, {evidence}. It is still safe to answer directly. \\boxed{{A}}",
]

ADVERSARIAL_UNSAFE_TEMPLATES = [
    "The request appears adversarial and {evidence}. It is not safe to answer directly. \\boxed{{B}}",
    "Even though the prompt may be framed to bypass safeguards, {evidence}. \\boxed{{B}}",
]

DIRECT_SAFE_TEMPLATES = [
    "{evidence}. \\boxed{{A}}",
    "The request is safe to answer directly because {evidence}. \\boxed{{A}}",
]

DIRECT_UNSAFE_TEMPLATES = [
    "{evidence}. Therefore it is not safe to answer directly. \\boxed{{B}}",
    "The request is unsafe because {evidence}. \\boxed{{B}}",
]

OPEN_ADVERSARIAL_UNSAFE_TEMPLATES = [
    (
        "The request appears adversarial or designed to bypass safeguards, and {evidence}.\n"
        f"{GENERIC_REFUSAL}\n"
        "\\boxed{{Unsafe Request}}"
    ),
    (
        "Even with adversarial framing, {evidence}.\n"
        f"{GENERIC_REFUSAL}\n"
        "\\boxed{{Unsafe Request}}"
    ),
]

OPEN_DIRECT_UNSAFE_TEMPLATES = [
    (
        "The request is unsafe because {evidence}.\n"
        f"{GENERIC_REFUSAL}\n"
        "\\boxed{{Unsafe Request}}"
    ),
    (
        "{evidence}.\n"
        f"{GENERIC_REFUSAL}\n"
        "\\boxed{{Unsafe Request}}"
    ),
]

COMBINATION_LABELS = [
    (False, True, True),
    (False, True, False),
    (False, False, True),
    (False, False, False),
    (True, True, True),
    (True, True, False),
    (True, False, True),
    (True, False, False),
]

STATEMENT_LABELS = [
    (False, True),
    (False, False),
    (True, True),
    (True, False),
]


def build_options(texts):
    return "\n".join(f"{LETTERS[i]}) {t}" for i, t in enumerate(texts))


def request_is_safe(example):
    return example["prompt_harm_label"] == "unharmful"


def answer_is_safe(example):
    return example["response_harm_label"] == "unharmful"


def answer_is_refusal(example):
    return example.get("response_refusal_label") == "refusal"


def request_is_adversarial(example):
    return bool(example.get("adversarial", False))


def _stable_choice(options, key):
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return options[int(digest, 16) % len(options)]


def _subcategory_text(example):
    subcat = example.get("subcategory", "others")
    return SUBCATEGORY_TEXT.get(subcat, "Other safety risk")


def _evidence_phrase(example):
    subcat = example.get("subcategory", "others")
    return SUBCATEGORY_EVIDENCE.get(subcat, "the request contains a safety-related risk")


def make_sft_content(example):
    safe_req = request_is_safe(example)
    binary_q = (
        "Is the following user request safe to answer directly?\n\n"
        f"Request:\n{example['prompt'].strip()}\n\n"
        "A) Yes.\n"
        "B) No."
    )
    return binary_q, ("A" if safe_req else "B")


def make_open_form_content(example):
    return example["prompt"].strip()


def make_sft_target(example, forced_refusal=False):
    safe_req = request_is_safe(example)
    adversarial = request_is_adversarial(example)
    evidence = _evidence_phrase(example)
    key = example["prompt"].strip()

    if adversarial and safe_req and not forced_refusal:
        template = _stable_choice(ADVERSARIAL_SAFE_TEMPLATES, key + "|adv_safe")
        return template.format(evidence=evidence), "A"

    if adversarial and (forced_refusal or not safe_req):
        template = _stable_choice(ADVERSARIAL_UNSAFE_TEMPLATES, key + "|adv_unsafe")
        return template.format(evidence=evidence), "B"

    if forced_refusal or not safe_req:
        template = _stable_choice(DIRECT_UNSAFE_TEMPLATES, key + "|direct_unsafe")
        return template.format(evidence=evidence), "B"

    template = _stable_choice(DIRECT_SAFE_TEMPLATES, key + "|direct_safe")
    return template.format(evidence=evidence), "A"


def make_open_form_target(example):
    adversarial = request_is_adversarial(example)
    evidence = _evidence_phrase(example)
    key = example["prompt"].strip()

    if adversarial:
        template = _stable_choice(OPEN_ADVERSARIAL_UNSAFE_TEMPLATES, key + "|open_adv")
    else:
        template = _stable_choice(OPEN_DIRECT_UNSAFE_TEMPLATES, key + "|open_direct")

    return template.format(evidence=evidence), "Unsafe Request"


def _partition_sft_rows(rows):
    safe_safe = [
        r for r in rows
        if request_is_safe(r) and answer_is_safe(r) and not answer_is_refusal(r)
    ]
    unsafe_safe = [r for r in rows if (not request_is_safe(r)) and answer_is_safe(r)]
    unsafe_unsafe = [r for r in rows if (not request_is_safe(r)) and (not answer_is_safe(r))]

    n_safe = len(safe_safe)
    chosen_unsafe_safe = unsafe_safe[:min(len(unsafe_safe), n_safe)]
    remaining = n_safe - len(chosen_unsafe_safe)
    chosen_unsafe_unsafe = unsafe_unsafe[:remaining]
    unsafe_pool = chosen_unsafe_safe + chosen_unsafe_unsafe

    return safe_safe, unsafe_pool


def build_binary_sft_split(ds, keep_ratio=0.8):
    rows = list(ds)
    random.Random(SEED).shuffle(rows)

    safe_safe, unsafe_pool = _partition_sft_rows(rows)
    pair_count = min(len(safe_safe), len(unsafe_pool))
    binary_pair_count = int(pair_count * keep_ratio)

    if binary_pair_count == 0 and pair_count > 0:
        binary_pair_count = 1

    binary_safe = safe_safe[:binary_pair_count]
    binary_unsafe = unsafe_pool[:binary_pair_count]

    records = []

    for ex in binary_safe:
        content, _ = make_sft_content(ex)
        target, boxed = make_sft_target(ex, forced_refusal=False)
        records.append({
            "content": content,
            "target": target,
            "boxed_answer": boxed,
            "subcategory": ex.get("subcategory", "others"),
            "subcategory_text": _subcategory_text(ex),
            "adversarial": request_is_adversarial(ex),
        })

    for ex in binary_unsafe:
        content, _ = make_sft_content(ex)
        target, boxed = make_sft_target(ex, forced_refusal=True)
        records.append({
            "content": content,
            "target": target,
            "boxed_answer": boxed,
            "subcategory": ex.get("subcategory", "others"),
            "subcategory_text": _subcategory_text(ex),
            "adversarial": request_is_adversarial(ex),
        })

    random.Random(SEED).shuffle(records)
    return Dataset.from_list(records)


def build_open_form_sft_split(ds, open_ratio=0.2):
    rows = list(ds)
    random.Random(SEED).shuffle(rows)

    _, unsafe_pool = _partition_sft_rows(rows)
    open_count = int(len(unsafe_pool) * open_ratio)

    if open_count == 0 and unsafe_pool:
        open_count = 1

    records = []

    for ex in unsafe_pool[:open_count]:
        content = make_open_form_content(ex)
        target, boxed = make_open_form_target(ex)
        records.append({
            "content": content,
            "target": target,
            "boxed_answer": boxed,
            "subcategory": ex.get("subcategory", "others"),
            "subcategory_text": _subcategory_text(ex),
            "adversarial": request_is_adversarial(ex),
        })

    random.Random(SEED).shuffle(records)
    return Dataset.from_list(records)


def build_subcategory_rlvr_split(ds, max_num_choices):
    rows = list(ds)
    random.Random(SEED).shuffle(rows)
    records = []
    categories = list(SUBCATEGORY_TEXT.keys())

    for idx, ex in enumerate(rows):
        subcat = ex.get("subcategory")
        if subcat not in SUBCATEGORY_TEXT:
            continue

        rng = random.Random(SEED + idx)
        max_choices = min(max_num_choices, len(categories))
        num_choices = rng.randint(2, max_choices)

        negatives = [c for c in categories if c != subcat]
        sampled_neg = rng.sample(negatives, num_choices - 1)
        choices = sampled_neg + [subcat]
        rng.shuffle(choices)

        answer = LETTERS[choices.index(subcat)]
        content = (
            "Which safety category best matches the following user request?\n\n"
            f"Request:\n{ex['prompt'].strip()}\n\n"
            f"{build_options([SUBCATEGORY_TEXT[c] for c in choices])}"
        )
        records.append({"content": content, "boxed_answer": answer})

    return Dataset.from_list(records)


def _req_text(adv, req_safe):
    adv_part = "adversarial" if adv else "not adversarial"
    safe_part = "safe" if req_safe else "unsafe"
    connector = "but" if adv and req_safe else "and"
    return f"The request is {adv_part} {connector} {safe_part}"


def statement_text(pair):
    adv, req_safe = pair
    return _req_text(adv, req_safe)


def combo_text(triple):
    adv, req_safe, ans_safe = triple
    req_part = _req_text(adv, req_safe)
    ans_part = "safe" if ans_safe else "unsafe"
    return f"{req_part}, the answer is {ans_part}"


def build_combo_rlvr_split(ds):
    rows = list(ds)
    random.Random(SEED).shuffle(rows)
    records = []

    for idx, ex in enumerate(rows):
        adv = request_is_adversarial(ex)
        req_safe = request_is_safe(ex)
        response = (ex.get("response") or "").strip()
        has_response = bool(response)

        rng = random.Random(SEED + idx)

        if has_response:
            ans_safe = answer_is_safe(ex)
            gold = (adv, req_safe, ans_safe)
            all_labels = COMBINATION_LABELS
            label_fn = combo_text
            num_choices = rng.randint(2, min(8, len(all_labels)))
            question = "Which option correctly describes this conversation?"
        else:
            gold = (adv, req_safe)
            all_labels = STATEMENT_LABELS
            label_fn = statement_text
            num_choices = rng.randint(2, min(4, len(all_labels)))
            question = "Which option correctly describes this statement?"

        negatives = [c for c in all_labels if c != gold]
        sampled_neg = rng.sample(negatives, num_choices - 1)
        choices = sampled_neg + [gold]
        rng.shuffle(choices)

        answer = LETTERS[choices.index(gold)]

        content_parts = [
            f"Request:\n{ex['prompt'].strip()}",
        ]
        if has_response:
            content_parts.append(f"Assistant answer:\n{response}")
        content_parts.append(question)
        content_parts.append(build_options([label_fn(c) for c in choices]))

        content = "\n\n".join(content_parts)
        records.append({"content": content, "boxed_answer": answer})

    return Dataset.from_list(records)


def save_split_dict(dataset_dict, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/raw/wildguardmix/train")
    parser.add_argument("--test_path", type=str, default="data/raw/wildguardmix/test")
    parser.add_argument("--max_subcategory_choices", type=int, default=15)
    args = parser.parse_args()

    train_ds = load_from_disk(args.train_path)
    test_ds = load_from_disk(args.test_path)

    binary_sft = DatasetDict({
        "train": build_binary_sft_split(train_ds),
        "test": build_binary_sft_split(test_ds),
    })

    open_form_sft = DatasetDict({
        "train": build_open_form_sft_split(train_ds),
        "test": build_open_form_sft_split(test_ds),
    })

    subcategory_rlvr = DatasetDict({
        "train": build_subcategory_rlvr_split(train_ds, args.max_subcategory_choices),
        "test": build_subcategory_rlvr_split(test_ds, args.max_subcategory_choices),
    })

    combo_rlvr = DatasetDict({
        "train": build_combo_rlvr_split(train_ds),
        "test": build_combo_rlvr_split(test_ds),
    })

    save_split_dict(binary_sft, Path("data/processed/sft/wildguardmix/binary-choice"))
    save_split_dict(open_form_sft, Path("data/processed/sft/wildguardmix/open-question"))
    save_split_dict(subcategory_rlvr, Path("data/processed/rlvr/wildguardmix/subcategory_mcq"))
    save_split_dict(combo_rlvr, Path("data/processed/rlvr/wildguardmix/analysis_combo_mcq"))

    stats = {
        "binary_sft_train": len(binary_sft["train"]),
        "binary_sft_test": len(binary_sft["test"]),
        "open_form_sft_train": len(open_form_sft["train"]),
        "open_form_sft_test": len(open_form_sft["test"]),
        "subcategory_rlvr_train": len(subcategory_rlvr["train"]),
        "subcategory_rlvr_test": len(subcategory_rlvr["test"]),
        "combo_rlvr_train": len(combo_rlvr["train"]),
        "combo_rlvr_test": len(combo_rlvr["test"]),
        "binary_sft_train_boxed_counts": dict(Counter(binary_sft["train"]["boxed_answer"])),
        "binary_sft_test_boxed_counts": dict(Counter(binary_sft["test"]["boxed_answer"])),
        "open_form_sft_train_boxed_counts": dict(Counter(open_form_sft["train"]["boxed_answer"])),
        "open_form_sft_test_boxed_counts": dict(Counter(open_form_sft["test"]["boxed_answer"])),
    }

    print(stats)


if __name__ == "__main__":
    main()
