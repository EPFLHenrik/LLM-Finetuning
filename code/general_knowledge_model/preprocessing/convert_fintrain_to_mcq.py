#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path


CHOICE_MARKER_RE = re.compile(r"(?<!\w)([A-Z])[\).]\s*")
ANSWER_RE = re.compile(r"\bAnswer:\s*([A-Z])\b", re.IGNORECASE)
NUMBERED_PREFIX_RE = re.compile(r"^\d+\.\s+")

# Applied to all raw source fields (question, choices, scenario, exhibit, justification)
BAD_PATTERNS = [
    re.compile(r"\bfigure \d+\b", re.IGNORECASE),
    re.compile(r"\bin exhibit\b", re.IGNORECASE),
    re.compile(r"\bexhibit\s*\d+\b", re.IGNORECASE),   # "Exhibit 2", "exhibit 3"
    re.compile(r"\bthe exhibit\b", re.IGNORECASE),
    re.compile(r"\b[A-Z][a-z]+'s\b"),
    re.compile(r"bond[,\s]*b\) bond", re.IGNORECASE),
]

# Applied only to the question stem (avoids false positives in GPT-4 justification text)
CONTEXT_PATTERNS = [
    re.compile(r"\bthese\s+(?:three|two|four|five|\d+)\s+\w+", re.IGNORECASE),  # "these three stocks"
    re.compile(r"\bgiven\s+these\b", re.IGNORECASE),       # "given these factors"
    re.compile(r"\bbased on\s+these\b", re.IGNORECASE),    # "based on these results"
    re.compile(r"\bfollowing\s+(?:table|chart|data|graph)\b", re.IGNORECASE),  # "following table"
]

NUMERIC_CHOICE_RE = re.compile(r"^[\$\-\+€£]?\d[\d,.\s%bps]*$")
STANDALONE_SHORT_RE = re.compile(
    r"^(?:yes|no|both|all|none|neither|true|false|[A-Z]\s+only)$",
    re.IGNORECASE,
)


def parse_choices(raw_choices: str):
    text = " ".join(part.strip() for part in raw_choices.splitlines() if part.strip())
    choices = []
    matches = list(CHOICE_MARKER_RE.finditer(text))
    for index, match in enumerate(matches):
        letter = match.group(1)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        choice_text = text[start:end].strip().rstrip(".,;:")
        if choice_text:
            choices.append((letter, choice_text))
    return choices


def has_truncated_choice(choices):
    """Reject records where a choice text is suspiciously short (likely cut off in source)."""
    for _, text in choices:
        stripped = text.strip()
        if (
            len(stripped) <= 3
            and not NUMERIC_CHOICE_RE.match(stripped)
            and not STANDALONE_SHORT_RE.match(stripped)
        ):
            return True
    return False


def is_numeric_without_data(question_stem: str, choices):
    """Reject questions whose choices are all numeric but the stem contains no numbers.

    These are calculation questions that need prior vignette data to answer.
    """
    if re.search(r"\d", question_stem):
        return False
    num_numeric = sum(1 for _, t in choices if NUMERIC_CHOICE_RE.match(t.strip()))
    return num_numeric >= 2


def build_question(question: str, choices):
    formatted_choices = ", ".join(f"{letter}) {text}" for letter, text in choices)
    return f"{question.strip()} {formatted_choices}".strip()


def extract_answer(obj, choices):
    gpt4_answer_justification = obj.get("gpt4_answer_justification")
    if isinstance(gpt4_answer_justification, str):
        match = ANSWER_RE.search(gpt4_answer_justification)
        if match:
            return match.group(1).upper()

    gpt4_answer = obj.get("gpt4_answer")
    if isinstance(gpt4_answer, str):
        match = ANSWER_RE.search(gpt4_answer)
        if match:
            return match.group(1).upper()

    answer = obj.get("answer")
    if isinstance(answer, str):
        stripped = answer.strip()
        if stripped and stripped.upper() != "N/A":
            if len(stripped) == 1 and stripped.isalpha():
                return stripped.upper()
            match = ANSWER_RE.search(stripped)
            if match:
                return match.group(1).upper()
            for letter, text in choices:
                if stripped == text or stripped.endswith(text):
                    return letter
            # Recover from verbose answers like "A. explanation..." or "B) reason..."
            lead = re.match(r"^([A-Z])[.\)]\s", stripped)
            if lead:
                return lead.group(1).upper()

    return None


def should_skip_record(obj):
    question_text = str(obj.get("questions", ""))
    all_fields_text = "\n".join([
        question_text,
        str(obj.get("answer_choices", "")),
        str(obj.get("scenario", "")),
        str(obj.get("exhibit", "")),
        str(obj.get("gpt4_answer_justification", "")),
    ])
    if any(p.search(all_fields_text) for p in BAD_PATTERNS):
        return True
    if any(p.search(question_text) for p in CONTEXT_PATTERNS):
        return True
    return False


def convert_record(obj):
    if should_skip_record(obj):
        return None

    raw_choices = obj.get("answer_choices")
    if not isinstance(raw_choices, str) or raw_choices.strip() in {"", "N/A"}:
        return None

    choices = parse_choices(raw_choices)
    if not choices:
        return None

    if has_truncated_choice(choices):
        return None

    answer = extract_answer(obj, choices)
    if answer is None:
        return None

    # Answer must be a single letter present among the parsed choices
    choice_letters = {letter for letter, _ in choices}
    if len(answer) != 1 or not answer.isalpha() or answer not in choice_letters:
        return None

    # Strip leading question number artifact (e.g. "1. ", "3. ")
    question_stem = NUMBERED_PREFIX_RE.sub("", obj.get("questions", "").strip())

    if is_numeric_without_data(question_stem, choices):
        return None

    return {
        "prompt": build_question(question_stem, choices),
        "answer": answer,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert FinTrain JSONL to filtered MCQ JSONL")
    parser.add_argument("input", nargs="?", default="data/datasets/train_datasets/FinTrain/train.jsonl")
    parser.add_argument("output", nargs="?", default="data/preprocessing/FinTrain/train.jsonl")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0
    seen_questions: set[str] = set()

    with input_path.open("r", encoding="utf-8") as infile, output_path.open("w", encoding="utf-8") as outfile:
        for line in infile:
            total += 1
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            converted = convert_record(obj)
            if converted is None:
                continue
            # Deduplicate on exact question text
            if converted["prompt"] in seen_questions:
                continue
            seen_questions.add(converted["prompt"])
            kept += 1
            outfile.write(json.dumps(converted, ensure_ascii=False) + "\n")

    print(f"Converted {kept} / {total} records to {output_path}")


if __name__ == "__main__":
    main()
